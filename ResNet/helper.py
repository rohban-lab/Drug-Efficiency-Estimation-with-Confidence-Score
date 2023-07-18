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
from model import Conv
from  dataset import *
from torch.nn.modules.batchnorm import _BatchNorm
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
DEVICE='cuda:0'
#train network without Confidence
def train(model, train_loader, val_loader, optimizer,Loss,mode, num_epochs=10):

    train_log = []
    plt.figure(figsize=(16, 8))
    final_out=torch.tensor([])
    for epoch in range(1, num_epochs+1):
         
        train_loss = []
        model.train()
        for inputs, targets  in tqdm(train_loader, desc='Training epoch ' + str(epoch), leave=False):        
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)        
            optimizer.zero_grad()
            pred= model(inputs)
            loss=Loss(pred,targets)*10
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

        train_log.append(np.mean(train_loss))
        
        torch.save(model.state_dict(), "models/res18_Score_"+mode+".pth")

        plt.title(f'{mode}')
        plt.xlabel('epoch') 
        plt.ylabel('loss') 
        plt.plot(range(1, epoch+1), train_log, color='C0', label='Train')
        plt.savefig("plots/"+mode+'.png')
def plot_(train_log,log_name,mode,epoch):
    plt.figure(figsize=(16, 8))
    plt.title(mode+"_"+log_name)
    plt.xlabel('epoch') 
    plt.ylabel('loss') 
    plt.plot(range(1, epoch+1), train_log, color='C0', label='Train')
    plt.savefig("plots/"+mode+"_"+log_name+'.png')
    plt.close()
#train network with Confidence
def train_confidence(model, train_loader, val_loader, optimizer,Loss,mode, num_epochs=10):

    train_total_loss_log = []
    train_reg_loss_log =[]
    train_confidence_loss_log =[]
    train_confidence_loss_lmbda_log=[]

    final_out=torch.tensor([])
    lmbda=0.2
    budget=0.35
    for epoch in range(1, num_epochs+1):
         
        train_total_loss = []
        train_reg_loss =[]
        train_confidence_loss =[]
        train_confidence_loss_lmbda=[]
        model.train()
        counter=0
        for inputs, targets  in tqdm(train_loader, desc='Training epoch ' + str(epoch), leave=False):        
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            pred_original,confidence= model(inputs)
            b = torch.bernoulli(torch.Tensor(confidence.size()).uniform_(0, 1)).cuda()
            conf = confidence * b + (1 - b)
            pred_new = pred_original * conf.expand_as(pred_original) + targets * (1 - conf.expand_as(targets))
            confidence_loss = torch.mean(-torch.log(confidence))
            reg_loss=Loss(pred_new,targets)
            total_loss = reg_loss + (lmbda * confidence_loss)
            total_loss.backward()
            optimizer.step()
            train_total_loss.append(total_loss.item())
            train_reg_loss.append(reg_loss.item())
            train_confidence_loss.append(confidence_loss.item())
            train_confidence_loss_lmbda.append(lmbda*confidence_loss.item())
            counter+=1
            if counter%5==0:
              if budget > confidence_loss.item():
                lmbda = lmbda / 1.05
              elif budget <= confidence_loss.item():
                lmbda = lmbda / 0.97
        train_total_loss_log.append(np.mean(train_total_loss))
        train_reg_loss_log.append(np.mean(train_reg_loss))
        train_confidence_loss_log.append(np.mean(train_confidence_loss))
        train_confidence_loss_lmbda_log.append(np.mean(train_confidence_loss_lmbda))
        torch.save(model.state_dict(), "models/res18_Score_conf"+mode+".pth")
        plot_(train_total_loss_log,'train_total_loss_log',mode,epoch)
        plot_(train_reg_loss_log,'train_reg_loss_log',mode,epoch)
        plot_(train_confidence_loss_log,'train_confidence_loss_log',mode,epoch)
        plot_(train_confidence_loss_lmbda_log,'train_confidence_loss_lmbda_log',mode,epoch)
#train network without Confidence and weakly approach
def train_confidence_supervised(model, train_loader, val_loader, optimizer,Loss,mode, num_epochs=10):

    train_total_loss_log = []
    train_reg_loss_log =[]
    train_confidence_loss_log =[]
    train_confidence_loss_lmbda_log=[]

    final_out=torch.tensor([])
    lmbda=0.2
    budget=0.08
    for epoch in range(1, num_epochs+1):
         
        train_total_loss = []
        train_reg_loss =[]
        train_confidence_loss =[]
        train_confidence_loss_lmbda=[]
        model.train()
        counter=0
        for inputs, targets  in tqdm(train_loader, desc='Training epoch ' + str(epoch), leave=False):        
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            pred_original,confidence= model(inputs)
            b = torch.bernoulli(torch.Tensor(confidence.size()).uniform_(0, 1)).cuda()
            conf = confidence * b + (1 - b)
            pred_new = pred_original * conf.expand_as(pred_original) + targets * (1 - conf.expand_as(targets))
            confidence_loss = torch.mean(-torch.log(confidence))
            reg_loss=Loss(pred_new,targets)
            total_loss = reg_loss + (lmbda * confidence_loss)
            total_loss.backward()
            optimizer.step()
            train_total_loss.append(total_loss.item())
            train_reg_loss.append(reg_loss.item())
            train_confidence_loss.append(confidence_loss.item())
            train_confidence_loss_lmbda.append(lmbda*confidence_loss.item())
            #counter+=1
            #if counter%5==0:
            if budget > confidence_loss.item():
                lmbda = lmbda / 1.01
            elif budget <= confidence_loss.item():
                lmbda = lmbda / 0.99
        train_total_loss_log.append(np.mean(train_total_loss))
        train_reg_loss_log.append(np.mean(train_reg_loss))
        train_confidence_loss_log.append(np.mean(train_confidence_loss))
        train_confidence_loss_lmbda_log.append(np.mean(train_confidence_loss_lmbda))
        torch.save(model.state_dict(), "models/res18_Score_conf"+mode+".pth")
        plot_(train_total_loss_log,'train_total_loss_log',mode,epoch)
        plot_(train_reg_loss_log,'train_reg_loss_log',mode,epoch)
        plot_(train_confidence_loss_log,'train_confidence_loss_log',mode,epoch)
        plot_(train_confidence_loss_lmbda_log,'train_confidence_loss_lmbda_log',mode,epoch)

# print just prediction and confidence score for one batch
def confidence_val(model, train_loader, val_loader, optimizer,Loss, num_epochs=10):

    for epoch in range(1, num_epochs+1):
     
        model.eval()
        for inputs, targets  in tqdm(train_loader, desc='Training epoch ' + str(epoch), leave=False):        
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            pred_original,confidence= model(inputs)
            print(pred_original)
            print(confidence)
            exit()
# print just prediction for one batch
def val(model, train_loader, val_loader, optimizer,Loss, num_epochs=10):

    for epoch in range(1, num_epochs+1):
     
        model.eval()
        for inputs, targets  in tqdm(train_loader, desc='Training epoch ' + str(epoch), leave=False):        
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            pred_original= model(inputs)
            print(pred_original)
            exit()
# return  logit for calibration analyses
def calibration_eval(model, train_loader, val_loader, optimizer,Loss, num_epochs=10):
        model.eval()
        dict_con={}
        epoch=0
        for i in range(10):
           dict_con[i]=[]
        for inputs, targets  in tqdm(train_loader, desc='Training epoch ' + str(epoch), leave=False):        
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            pred_original,confidence= model(inputs)
            for i in range(pred_original.shape[0]):
                dict_con[int((pred_original[i,0].detach().item()*100)//10)].append(confidence[i].detach().item())
        return dict_con
'''
def test_calibration1(model, train_loader, val_loader, optimizer,Loss, num_epochs=10):
     
        model.eval()
        sum_=0
        epoch=0
        count_=0
        for inputs, targets  in tqdm(train_loader, desc='Training epoch ' + str(epoch), leave=False):        
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            pred_original,confidence= model(inputs)
            print(pred_original)
            for i in range(pred_original.shape[0]):
                sum_+=(pred_original[i,0].item()-1)**2
            count_+=pred_original.shape[0]
        print(sum_)
        print(count_)
'''
# Computing drug metrics for all images
def test(model, train_loader, val_loader, optimizer,Loss, num_epochs=10):
        final_out=torch.tensor([], device='cuda')
        test_loss = []
        model.eval()

        df = pd.DataFrame([], columns=['experiment','treatment','treatment_con','plate','well','site1','site2','site3','site4','site1_val','site2_val','site3_val','site4_val','well_median'])

        with torch.no_grad():  

            for inputs, treatment,treatment_con,experiment,plate,well,site1,site2,site3,site4 in tqdm(val_loader, desc='Validation', leave=False):         
                out_put=[]
                out_put1=[]
                well_mean=[]
                preds=[]
                treatment = (treatment[0].replace('/', '1'),)
                for i in range(0,len(inputs)): 
                    input_=inputs[i]
                    input_=torch.squeeze(input_,0)
                    if input_.shape[0]!=0:
                        input_ = input_.to(DEVICE)
                        pred = model(input_)
                        preds.append(pred.shape[0])
                        final_out=torch.tensor([], device='cuda')
                        final_out=torch.cat((final_out,pred[:, 0]),0)
                        out_put1.append(final_out)
                        out_put.append(final_out)
                        well_mean.append(torch.median(final_out).item())
                    else:
                        out_put1.append(torch.tensor([-1]))
                new_row = {'experiment':experiment[0],'treatment':treatment[0],'treatment_con':treatment_con[0], "plate":plate[0], 'well':well[0], 'site1':site1[0],'site2':site2[0],
                           'site3':site3[0],'site4':site4[0],
                           'site1_val':torch.median(out_put1[0]).item(),'site2_val':torch.median(out_put1[1]).item(),'site3_val':torch.median(out_put1[2]).item(),
                            'site4_val':torch.median(out_put1[3]).item(),'well_median':np.median(well_mean)}
                df = df.append(new_row, ignore_index=True)
                '''
                lens=len(preds)
                fig, axs = plt.subplots(2, 2)
                fig.suptitle(treatment[0]+"  "+str(treatment_con[0])+" _ "+str(well[0])+" _ "+str(plate[0]))
                if lens > 0 : 
                    axs[0, 0].hist(out_put[0].tolist() , bins=[0, 0.1 , 0.2, 0.3, 0.4, 0.5, 0.6,0.7,0.8,0.9,1])
                    axs[0, 0].set_title('Site 1 _'+str(preds[0]))

                if lens >1:
                    axs[0, 1].hist(out_put[1].tolist(), bins=[0, 0.1 , 0.2, 0.3, 0.4, 0.5, 0.6,0.7,0.8,0.9,1])
                    axs[0, 1].set_title('Site 2 _'+str(preds[1]))
                if lens >2:

                    axs[1, 0].hist(out_put[2].tolist(), bins=[0, 0.1 , 0.2, 0.3, 0.4, 0.5, 0.6,0.7,0.8,0.9,1])
                    axs[1, 0].set_title('Site 3 _'+str(preds[2]))
                if lens >3:
                    axs[1, 1].hist(out_put[3].tolist(),bins=[0, 0.1 , 0.2, 0.3, 0.4, 0.5, 0.6,0.7,0.8,0.9,1])
                    axs[1, 1].set_title('Site 4 _'+str(preds[3]))
                plt.tight_layout()
                fig.savefig('Results/'+treatment[0]+"_"+treatment_con[0]+"_"+plate[0]+"_"+well[0]+"_"+experiment[0]+".png")
                '''
            df.to_csv("CSV/mock_TOTAL_super_mse.csv")
# Computing drug metrics with Confidence for all images
def test_conf(model, train_loader, val_loader, optimizer,Loss, num_epochs=10):
        final_out=torch.tensor([], device='cuda')
        test_loss = []
        model.eval()

        df = pd.DataFrame([], columns=['experiment','treatment','treatment_con','plate','well','site1','site2','site3','site4','site1_val','site2_val','site3_val','site4_val','site1_conf','site2_conf','site3_conf','site4_conf','well_median'])

        with torch.no_grad():  

            for inputs, treatment,treatment_con,experiment,plate,well,site1,site2,site3,site4 in tqdm(val_loader, desc='Validation', leave=False):         
                out_put=[]
                out_put1=[]
                well_mean=[]
                site_conf=[]
                preds=[]
                treatment = (treatment[0].replace('/', '1'),)
                for i in range(0,len(inputs)): 
                    input_=inputs[i]
                    input_=torch.squeeze(input_,0)
                    if input_.shape[0]!=0:
                        input_ = input_.to(DEVICE)
                        pred,conf = model(input_)
                        site_conf.append(np.median(conf.cpu()))
                        preds.append(pred.shape[0])
                        final_out=torch.tensor([], device='cuda')
                        final_out=torch.cat((final_out,pred[:, 0]),0)
                        out_put1.append(final_out)
                        out_put.append(final_out)
                        well_mean.append(torch.median(final_out).item())
                    else:
                        out_put1.append(torch.tensor([-1]))
                        site_conf.append(-1)
                new_row = {'experiment':experiment[0],'treatment':treatment[0],'treatment_con':treatment_con[0], "plate":plate[0], 'well':well[0], 'site1':site1[0],'site2':site2[0],
                           'site3':site3[0],'site4':site4[0],
                           'site1_val':torch.median(out_put1[0]).item(),'site2_val':torch.median(out_put1[1]).item(),'site3_val':torch.median(out_put1[2]).item(),
                            'site4_val':torch.median(out_put1[3]).item(),'site1_conf':site_conf[0],'site2_conf':site_conf[1],'site3_conf':site_conf[2],
                             'site4_conf':site_conf[3],'well_median':np.median(well_mean)}
                df = df.append(new_row, ignore_index=True)
                '''
                lens=len(preds)
                fig, axs = plt.subplots(2, 2)
                fig.suptitle(treatment[0]+"  "+str(treatment_con[0])+" _ "+str(well[0])+" _ "+str(plate[0]))
                if lens > 0 : 
                    axs[0, 0].hist(out_put[0].tolist() , bins=[0, 0.1 , 0.2, 0.3, 0.4, 0.5, 0.6,0.7,0.8,0.9,1])
                    axs[0, 0].set_title('Site 1 _'+str(preds[0]))

                if lens >1:
                    axs[0, 1].hist(out_put[1].tolist(), bins=[0, 0.1 , 0.2, 0.3, 0.4, 0.5, 0.6,0.7,0.8,0.9,1])
                    axs[0, 1].set_title('Site 2 _'+str(preds[1]))
                if lens >2:

                    axs[1, 0].hist(out_put[2].tolist(), bins=[0, 0.1 , 0.2, 0.3, 0.4, 0.5, 0.6,0.7,0.8,0.9,1])
                    axs[1, 0].set_title('Site 3 _'+str(preds[2]))
                if lens >3:
                    axs[1, 1].hist(out_put[3].tolist(),bins=[0, 0.1 , 0.2, 0.3, 0.4, 0.5, 0.6,0.7,0.8,0.9,1])
                    axs[1, 1].set_title('Site 4 _'+str(preds[3]))
                plt.tight_layout()
                fig.savefig('Results/'+treatment[0]+"_"+treatment_con[0]+"_"+plate[0]+"_"+well[0]+"_"+experiment[0]+".png")
                '''
            df.to_csv("CSV/mock_score_res_super_conf_0.2.csv")
#query for each well separately
def test_query(model, test_dataset,treatment,treatment_con,experiment,plate,well,count_l):
        final_out=torch.tensor([], device='cuda')
        test_loss = []
        model.eval()

        with torch.no_grad():  

                out_put=[]
                out_put1=[]
                well_mean=[]
                preds=[]
                inputs=test_dataset.callable(treatment,experiment,plate,well,count_l)
                treatment = treatment.replace('/', '1')
                for i in range(0,len(inputs)): 
                    input_=inputs[i]
                    input_=torch.squeeze(input_,0)
                    if input_.shape[0]!=0:
                        input_ = input_.to(DEVICE)
                        pred = model(input_)
                        preds.append(pred.shape[0])
                        final_out=torch.tensor([], device='cuda')
                        final_out=torch.cat((final_out,pred[:, 0]),0)
                        out_put1.append(final_out)
                        out_put.append(final_out)
                        well_mean.append(torch.median(final_out).item())
                    else:
                        out_put1.append(torch.tensor([-1]))
                lens=len(preds)
                fig, axs = plt.subplots(2, 2)
                fig.suptitle(treatment+"  "+str(treatment_con)+" _ "+str(well)+" _ "+str(plate))
                if lens > 0 : 
                    axs[0, 0].hist(out_put[0].tolist() , bins=[0, 0.1 , 0.2, 0.3, 0.4, 0.5, 0.6,0.7,0.8,0.9,1])
                    axs[0, 0].set_title('Site 1 _'+str(preds[0]))

                if lens >1:
                    axs[0, 1].hist(out_put[1].tolist(), bins=[0, 0.1 , 0.2, 0.3, 0.4, 0.5, 0.6,0.7,0.8,0.9,1])
                    axs[0, 1].set_title('Site 2 _'+str(preds[1]))
                if lens >2:

                    axs[1, 0].hist(out_put[2].tolist(), bins=[0, 0.1 , 0.2, 0.3, 0.4, 0.5, 0.6,0.7,0.8,0.9,1])
                    axs[1, 0].set_title('Site 3 _'+str(preds[2]))
                if lens >3:
                    axs[1, 1].hist(out_put[3].tolist(),bins=[0, 0.1 , 0.2, 0.3, 0.4, 0.5, 0.6,0.7,0.8,0.9,1])
                    axs[1, 1].set_title('Site 4 _'+str(preds[3]))
                plt.tight_layout()
                fig.savefig('loc_image/'+treatment+"_"+treatment_con+"_"+plate+"_"+well+"_"+experiment+".png")
                return out_put
'''
def test_query_confidence(model, test_dataset,treatment,treatment_con,experiment,plate,well,count_l):
        final_out=torch.tensor([], device='cuda')
        test_loss = []
        model.eval()

        with torch.no_grad():  

                out_put=[]
                out_put1=[]
                well_mean=[]
                preds=[]
                inputs=test_dataset.callable(treatment,experiment,plate,well,count_l)
                treatment = treatment.replace('/', '1')
                for i in range(0,len(inputs)): 
                    input_=inputs[i]
                    input_=torch.squeeze(input_,0)
                    if input_.shape[0]!=0:
                        input_ = input_.to(DEVICE)
                        pred,conf = model(input_)
                        print(conf)
                        print(pred)
                        print("---------")
                        preds.append(pred.shape[0])
                        final_out=torch.tensor([], device='cuda')
                        final_out=torch.cat((final_out,pred[:, 0]),0)
                        out_put1.append(final_out)
                        out_put.append(final_out)
                        well_mean.append(torch.median(final_out).item())
                    else:
                        out_put1.append(torch.tensor([-1]))
                lens=len(preds)
                fig, axs = plt.subplots(2, 2)
                fig.suptitle(treatment+"  "+str(treatment_con)+" _ "+str(well)+" _ "+str(plate))
                if lens > 0 : 
                    axs[0, 0].hist(out_put[0].tolist() , bins=[0, 0.1 , 0.2, 0.3, 0.4, 0.5, 0.6,0.7,0.8,0.9,1])
                    axs[0, 0].set_title('Site 1 _'+str(preds[0]))

                if lens >1:
                    axs[0, 1].hist(out_put[1].tolist(), bins=[0, 0.1 , 0.2, 0.3, 0.4, 0.5, 0.6,0.7,0.8,0.9,1])
                    axs[0, 1].set_title('Site 2 _'+str(preds[1]))
                if lens >2:

                    axs[1, 0].hist(out_put[2].tolist(), bins=[0, 0.1 , 0.2, 0.3, 0.4, 0.5, 0.6,0.7,0.8,0.9,1])
                    axs[1, 0].set_title('Site 3 _'+str(preds[2]))
                if lens >3:
                    axs[1, 1].hist(out_put[3].tolist(),bins=[0, 0.1 , 0.2, 0.3, 0.4, 0.5, 0.6,0.7,0.8,0.9,1])
                    axs[1, 1].set_title('Site 4 _'+str(preds[3]))
                plt.tight_layout()
                fig.savefig('loc_image/'+treatment+"_"+treatment_con+"_"+plate+"_"+well+"_"+experiment+".png")
                return out_put
'''
