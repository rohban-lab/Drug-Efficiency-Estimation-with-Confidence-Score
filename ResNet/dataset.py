from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from random import randint
from scipy.stats import beta
from collections import Counter
import pandas as pd
import numpy as np
import os
from PIL import Image
import torch
#Dataset Class for training the netwotk
class DrugdatasetNew(Dataset):

    def __init__(self,path, transform=None):
        self.input_dir=path+"/train"
        meta_data=pd.read_csv(path+"/train/cp_out_meta.csv")
        meta_data1=pd.read_csv(path+"/train/out_meta_vero_train.csv")
        meta_data=pd.concat([meta_data, meta_data1], ignore_index=True, sort=False)

        self.transform = transform
        self.cell_dict={}
        self.cell_dict_count={}
        self.len=0
        for i in meta_data["disease_condition"].unique():
            self.cell_dict[i]={}
            for j in meta_data["experiment"].unique():
                self.cell_dict[i][j]={}

            self.cell_dict_count[i]=0
        for index, row in meta_data.iterrows():
          if row["cell_count"]==0:
            print(row["cell_count"])
          if row["cell_count"]!=0:
            self.cell_dict_count[row["disease_condition"]]+=row["cell_count"]
            if row["plate"] not in self.cell_dict[row["disease_condition"]][row["experiment"]].keys():
                 self.cell_dict[row["disease_condition"]][row["experiment"]][row["plate"]]={}
                 self.cell_dict[row["disease_condition"]][row["experiment"]][row["plate"]][row["well"]]={1:0,2:0,3:0,4:0}
                 self.cell_dict[row["disease_condition"]][row["experiment"]][row["plate"]][row["well"]][row["site"]]=row["cell_count"]
            else:
                 if  row["well"] not in self.cell_dict[row["disease_condition"]][row["experiment"]][row["plate"]].keys():
                     self.cell_dict[row["disease_condition"]][row["experiment"]][row["plate"]][row["well"]]={1:0,2:0,3:0,4:0}
                 self.cell_dict[row["disease_condition"]][row["experiment"]][row["plate"]][row["well"]][row["site"]]=row["cell_count"]
        print(self.cell_dict)
        print(self.cell_dict_count)
        self.len=max(self.cell_dict_count["Active SARS-CoV-2"],self.cell_dict_count["Mock"])
    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        disease_condition=["Mock",'Active SARS-CoV-2']
        alpha=beta.rvs(2, 2, size=1).item()
        out=[]
        for item in disease_condition:
            keys=list(self.cell_dict[item].keys())
            experiment=keys[randint(0, len(keys)-1)]
            keys=list(self.cell_dict[item][experiment].keys())
            random_plate=keys[randint(0, len(keys)-1)]
            keys=list(self.cell_dict[item][experiment][random_plate].keys())
            random_well=keys[randint(0, len(keys)-1)]
            site=randint(1, 4)
            image_number=randint(1,self.cell_dict[item][experiment][random_plate][random_well][site])
            image_w=[]
            image_name=random_well+"_s"+str(site)

            for channel in range(1,6):
                image=image_name+"_w"+str(channel)+"_c"+str(image_number)+".png"
                image_w.append(Image.open(os.path.join(self.input_dir,experiment,"Plate"+str(random_plate),image)))
            image=torch.from_numpy(np.stack(image_w,0)).float()
            out.append(self.transform(image))
        return out[0]*alpha +  out[1]*(1-alpha),torch.tensor([alpha,1-alpha])

#Dataset Class for testing the netwotk
class DrugdatasetNew_test(Dataset):

    def __init__(self,path, transform=None):
        self.input_dir=path+"/train/"
        meta_data=pd.read_csv(path+"/train/cp_out_meta.csv")
        meta_data1=pd.read_csv(path+"/train/out_meta_vero_train.csv")
        meta_data=pd.concat([meta_data, meta_data1], ignore_index=True, sort=False)
        #meta_data=meta_data[meta_data["experiment"]=="HRCE-1"]

        self.transform = transform
        self.image_name=[]
        meta_data=meta_data[meta_data["disease_condition"]=="Mock"]
        #meta_data=meta_data[meta_data["disease_condition"]=="Active SARS-CoV-2"]
        #meta_data=meta_data[meta_data["disease_condition"]=="UV Inactivated SARS-CoV-2"]
        for index, row in meta_data.iterrows():
            image_n=row["well"]+"_s"+str(row["site"])
            for i in range(1,row["cell_count"]+1):
               self.image_name.append(image_n+"_c"+str(i)+"_"+str(row["plate"])+"_"+str(row["experiment"]))
        print(self.image_name)
        print(len(self.image_name))
    def __len__(self):
        return len(self.image_name)

    def __getitem__(self, idx):
        image_name=self.image_name[idx].split("_")
        image_w=[]
        for channel in range(1,6):
                image=image_name[0]+"_"+image_name[1]+"_w"+str(channel)+"_"+image_name[2]+".png"
                image_w.append(Image.open(os.path.join(self.input_dir,image_name[-1],"Plate"+str(image_name[3]),image)))
        image=torch.from_numpy(np.stack(image_w,0)).float()
        return self.transform(image),torch.tensor([1,0])
#Dataset Class for testing the netwotk without weakly supervised approaches
class DrugdatasetNew_supervised(Dataset):

    def __init__(self,path, transform=None):
        self.input_dir=path+"/train"
        meta_data=pd.read_csv(path+"/train/cp_out_meta.csv")
        meta_data1=pd.read_csv(path+"/train/out_meta_vero_train.csv")
        meta_data=pd.concat([meta_data, meta_data1], ignore_index=True, sort=False)

        self.transform = transform
        self.cell_dict={}
        self.cell_dict_count={}
        self.len=0
        for i in meta_data["disease_condition"].unique():
            self.cell_dict[i]={}
            for j in meta_data["experiment"].unique():
                self.cell_dict[i][j]={}

            self.cell_dict_count[i]=0
        for index, row in meta_data.iterrows():
          if row["cell_count"]==0:
            print(row["cell_count"])
          if row["cell_count"]!=0:
            self.cell_dict_count[row["disease_condition"]]+=row["cell_count"]
            if row["plate"] not in self.cell_dict[row["disease_condition"]][row["experiment"]].keys():
                 self.cell_dict[row["disease_condition"]][row["experiment"]][row["plate"]]={}
                 self.cell_dict[row["disease_condition"]][row["experiment"]][row["plate"]][row["well"]]={1:0,2:0,3:0,4:0}
                 self.cell_dict[row["disease_condition"]][row["experiment"]][row["plate"]][row["well"]][row["site"]]=row["cell_count"]
            else:
                 if  row["well"] not in self.cell_dict[row["disease_condition"]][row["experiment"]][row["plate"]].keys():
                     self.cell_dict[row["disease_condition"]][row["experiment"]][row["plate"]][row["well"]]={1:0,2:0,3:0,4:0}
                 self.cell_dict[row["disease_condition"]][row["experiment"]][row["plate"]][row["well"]][row["site"]]=row["cell_count"]
        print(self.cell_dict)
        print(self.cell_dict_count)
        self.len=max(self.cell_dict_count["Active SARS-CoV-2"],self.cell_dict_count["Mock"])
    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        disease_condition=["Mock",'Active SARS-CoV-2']
        cond_rand=randint(0, 1)
        item=disease_condition[cond_rand]
        keys=list(self.cell_dict[item].keys())
        experiment=keys[randint(0, len(keys)-1)]
        keys=list(self.cell_dict[item][experiment].keys())
        random_plate=keys[randint(0, len(keys)-1)]
        keys=list(self.cell_dict[item][experiment][random_plate].keys())
        random_well=keys[randint(0, len(keys)-1)]
        site=randint(1, 4)
        image_number=randint(1,self.cell_dict[item][experiment][random_plate][random_well][site])
        image_w=[]
        image_name=random_well+"_s"+str(site)
        if cond_rand==0:
           label=torch.tensor([1.0,0.0])
        else: 
           label=torch.tensor([0.0,1.0])
        for channel in range(1,6):
                image=image_name+"_w"+str(channel)+"_c"+str(image_number)+".png"
                image_w.append(Image.open(os.path.join(self.input_dir,experiment,"Plate"+str(random_plate),image)))
        image=torch.from_numpy(np.stack(image_w,0)).float()
        return self.transform(image),label


#Dataset Class for creating a dataset with each image correspond to one image in a specific dose 
#you should set batch size for this dataloader to one as itself try to load each well in one batch
class DrugdatasetNew_drug(Dataset):

    def __init__(self,path, transform=None):
        self.input_dir=path
        self.meta_data=pd.read_csv(path+"out_drug_train_tt.csv")
        #self.meta_data=self.meta_data[(self.meta_data["experiment"]=="HRCE-2")|(self.meta_data["experiment"]=="HRCE-1")]
        self.dict_image={}
        self.treatment=self.meta_data["treatment"].unique()
        self.current_index=0
        self.count=len(self.meta_data)//4
        for treat in self.treatment:
            meta_data=self.meta_data[self.meta_data["treatment"]==treat]
            meta_group=meta_data.groupby(["experiment","plate","well"])
            self.dict_image[treat]=set()
            for index, meta in meta_group:
                temp={}
                image_n=str(meta["treatment_conc"].unique()[0])+"_"+meta["experiment"].unique()[0]+"_"+str(meta["plate"].unique()[0])+"_"+meta["well"].unique()[0]
                for ind,row in meta.iterrows():
                    temp[row["site"]]=str(row["cell_count"])
                for i in range(1,5):
                    image_n+="_"+temp[i]

                self.dict_image[treat].add(image_n)
        self.transform = transform

        print(self.dict_image)
        print(len(self.meta_data))
    def __len__(self):
        return self.count

    def __getitem__(self, idx):
        treatment=self.treatment[self.current_index]
        #treatment='Ciclopirox'
        if len(self.dict_image[treatment])==0:
            #exit()
            self.current_index+=1
            treatment=self.treatment[self.current_index]
        image_data=self.dict_image[treatment].pop()
        split=image_data.split("_")
        out_put=[]
        print(split)
        for site in range(1,5):
         image_l=[]
         for count in range(1,int(split[site+3])+1):
           image_w=[]
           for channel in range(1,6):
                  image=split[3]+"_s"+str(site)+"_w"+str(channel)+"_c"+str(count)+".png"
                  image_w.append(Image.open(os.path.join(self.input_dir,split[1],"Plate"+split[2],image)))
           image=self.transform(torch.from_numpy(np.stack(image_w,0)).float())
          #print(image.shape)
          #print("-------")
           image_l.append(image)
         if len(image_l)==0:
            out_put.append(torch.Tensor([]))
         else:
            out_put.append(torch.stack(image_l, dim=0))
        return out_put,treatment,split[0],split[1],split[2],split[3],split[4],split[5],split[6],split[7]
#dataset object that return information of one Sspecific well
class DrugdatasetNew_query(Dataset):

    def __init__(self,path, transform=None):
        self.input_dir=path
        self.meta_data=pd.read_csv(path+"out_drug_train_tt.csv")
        #self.meta_data=self.meta_data[(self.meta_data["experiment"]=="HRCE-2")|(self.meta_data["experiment"]=="HRCE-1")]
        self.dict_image={}
        self.treatment=self.meta_data["treatment"].unique()
        self.current_index=0
        self.count=len(self.meta_data)//4
        for treat in self.treatment:
            meta_data=self.meta_data[self.meta_data["treatment"]==treat]
            meta_group=meta_data.groupby(["experiment","plate","well"])
            self.dict_image[treat]=set()
            for index, meta in meta_group:
                temp={}
                image_n=str(meta["treatment_conc"].unique()[0])+"_"+meta["experiment"].unique()[0]+"_"+str(meta["plate"].unique()[0])+"_"+meta["well"].unique()[0]
                for ind,row in meta.iterrows():
                    temp[row["site"]]=str(row["cell_count"])
                for i in range(1,5):
                    image_n+="_"+temp[i]
                self.dict_image[treat].add(image_n)
        self.transform = transform

        print(self.dict_image)
        print(len(self.meta_data))
    def __len__(self):
        return self.count

    def __getitem__(self, idx):
        print(i)

    def callable(self, treatment,experiment,Plate,well,count_l):
        out_put=[]
        for site in range(1,5):
         image_l=[]
         for count in range(1,int(count_l[site-1])+1):
           image_w=[]
           for channel in range(1,6):
                  image=well+"_s"+str(site)+"_w"+str(channel)+"_c"+str(count)+".png"
                  image_w.append(Image.open(os.path.join(self.input_dir,experiment,"Plate"+str(Plate),image)))
           image=self.transform(torch.from_numpy(np.stack(image_w,0)).float())
           image_l.append(image)
         if len(image_l)==0:
            out_put.append(torch.Tensor([]))
         else:
            out_put.append(torch.stack(image_l, dim=0))
        return out_put


















