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

class Conv(nn.Module):
    def __init__(self, input_shape, num_classes, convs, fcs, conv_drop_rate=None, fc_drop_rate=None):
        """
        Args:
            input_shape: the dimension of the input image.
            num_classes: number of output classes.
            convs: it's 'resnet18' or a list of tuples with these info\:
                (out_channels, kernel_size, stride, padding, has_pooling), the first four are
                the configuration for creating a Conv2d layer and if has_pooling was True, you
                must add a MaxPool2d after the Conv2d.
            fcs: a list of integers representing number of Linear neurons in each layer
            conv_drop_rate: float(0-1), drop rate used for Conv2d layers
            fc_drop_rate: float(0-1), drop rate used for Linear layers
        """
        super(ConvClassifier, self).__init__()
        self.conv_layers = None
        if convs == 'resnet18':
            list_l=[]
            list_l=[torch.nn.Conv2d(5,3,3,padding=0),nn.BatchNorm2d(3),nn.ReLU(True)]
            self.preproc=nn.Sequential(*list_l)
            self.conv_layers = torchvision.models.resnet18(pretrained=True)
            self.conv_layers=torch.nn.Sequential(*(list(self.conv_layers.children())[:-1]))
        elif type(convs) is list:
            conv_layers = []
            for index in range(len(convs)):
               out_channels, kernel_size, stride, padding, has_pooling=convs[index]
               if index==0:
                  conv_layers.append(nn.Conv2d(5,out_channels,kernel_size,stride,padding))
               else:
                  conv_layers.append(nn.Conv2d(convs[index-1][0],out_channels,kernel_size,stride,padding))
               conv_layers.append(nn.BatchNorm2d(out_channels))
               conv_layers.append(nn.ReLU(True))
               if conv_drop_rate!=None:
                  conv_layers.append(nn.Dropout(conv_drop_rate))
               if has_pooling:
                  conv_layers.append(nn.MaxPool2d(2, 2))                 

            self.conv_layers = nn.Sequential(*conv_layers)
        else:
            raise Exception(f'Wrong value for parameter `convs`: {convs}')
        # Get the output shape of last conv layer
        convs_output_shape = self.conv_layers(torch.randn(1, *input_shape)).shape[1:]
        print(convs_output_shape)
        self.embed_size = int(convs_output_shape[0])
        # Fully Connected Layers
        fc_layers = []

        
        fc_layers+=[nn.Flatten()]
        convs_output=int(torch.prod(torch.Tensor(list(convs_output_shape))).item())
        for index in range(len(fcs)):
          if index==0:
             fc_layers.append(nn.Linear(convs_output,fcs[index]))
          else:
             fc_layers.append(nn.Linear(fcs[index-1],fcs[index]))
          if index!=len(fcs)-1:
             fc_layers.append(nn.BatchNorm1d(fcs[index]))
             fc_layers.append(nn.ReLU(True))
             if fc_drop_rate!=None:
                fc_layers.append(nn.Dropout(p=fc_drop_rate))

          else:
            fc_layers.append(nn.Softmax(dim=1))
        self.fc_layers = nn.Sequential(*fc_layers)

        self.cross_entropy = nn.CrossEntropyLoss()
        
    def get_embed_size(self):
      return self.embed_size
    def forward(self, x):
        output=None
        out=self.preproc(x)
        output_conv = self.conv_layers(out)
        output=self.fc_layers(output_conv)

        return output
    def forward_eval(self,x):
        output_conv = self.conv_layers(self.preproc(x))
        output=self.fc_layers(output_conv)
        return output

class Conv_confidence(nn.Module):
    def __init__(self, input_shape, num_classes, convs, fcs, conv_drop_rate=None, fc_drop_rate=None):
        """
        Args:
            input_shape: the dimension of the input image.
            num_classes: number of output classes.
            convs: it's 'resnet18' or a list of tuples with these info\:
                (out_channels, kernel_size, stride, padding, has_pooling), the first four are
                the configuration for creating a Conv2d layer and if has_pooling was True, you
                must add a MaxPool2d after the Conv2d.
            fcs: a list of integers representing number of Linear neurons in each layer
            conv_drop_rate: float(0-1), drop rate used for Conv2d layers
            fc_drop_rate: float(0-1), drop rate used for Linear layers
        """
        super(ConvClassifier_confidence, self).__init__()
        self.conv_layers = None
        if convs == 'resnet18':
            list_l=[]
            list_l=[torch.nn.Conv2d(5,3,3,padding=0),nn.BatchNorm2d(3),nn.ReLU(True)]
            self.preproc=nn.Sequential(*list_l)
            self.conv_layers = torchvision.models.resnet18(pretrained=True)
            self.conv_layers=torch.nn.Sequential(*(list(self.conv_layers.children())[:-1]))
        elif type(convs) is list:
            conv_layers = []
            for index in range(len(convs)):
               out_channels, kernel_size, stride, padding, has_pooling=convs[index]
               if index==0:
                  conv_layers.append(nn.Conv2d(5,out_channels,kernel_size,stride,padding))
               else:
                  conv_layers.append(nn.Conv2d(convs[index-1][0],out_channels,kernel_size,stride,padding))
               conv_layers.append(nn.BatchNorm2d(out_channels))
               conv_layers.append(nn.ReLU(True))
               if conv_drop_rate!=None:
                  conv_layers.append(nn.Dropout(conv_drop_rate))
               if has_pooling:
                  conv_layers.append(nn.MaxPool2d(2, 2))                 

            self.conv_layers = nn.Sequential(*conv_layers)
        else:
            raise Exception(f'Wrong value for parameter `convs`: {convs}')
        # Get the output shape of last conv layer
        convs_output_shape = self.conv_layers(torch.randn(1, *input_shape)).shape[1:]
        print(convs_output_shape)
        self.embed_size = int(convs_output_shape[0])
        # Fully Connected Layers
        fc_layers = []
        fc_confidence =[]
        
        fc_layers+=[nn.Flatten()]
        fc_confidence+=[nn.Flatten()]
        convs_output=int(torch.prod(torch.Tensor(list(convs_output_shape))).item())
        for index in range(len(fcs)):
          if index==0:
             fc_layers.append(nn.Linear(convs_output,fcs[index]))
          else:
             fc_layers.append(nn.Linear(fcs[index-1],fcs[index]))
          if index!=len(fcs)-1:
             fc_layers.append(nn.BatchNorm1d(fcs[index]))
             fc_layers.append(nn.ReLU(True))
             if fc_drop_rate!=None:
                fc_layers.append(nn.Dropout(p=fc_drop_rate))

          else:
            fc_layers.append(nn.Softmax(dim=1))
        fc_confidence.append(nn.Linear(convs_output,1))
        fc_confidence.append(nn.Sigmoid())
        self.fc_layers = nn.Sequential(*fc_layers)
        self.fc_confidence = nn.Sequential(*fc_confidence)
        self.cross_entropy = nn.CrossEntropyLoss()

        
    def get_embed_size(self):
      return self.embed_size
    def forward(self, x):
        output=None
        out=self.preproc(x)
        output_conv = self.conv_layers(out)
        output=self.fc_layers(output_conv)
        confidence=self.fc_confidence(output_conv)
        return output,confidence
    def forward_eval(self,x):
        output_conv = self.conv_layers(self.preproc(x))
        output=self.fc_layers(output_conv)
        return output
