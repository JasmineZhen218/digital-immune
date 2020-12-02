import openslide
import os
import pickle
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.utils.data as data_utils
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

from skimage.filters import threshold_otsu
from skimage.color import rgb2gray
from skimage import measure
from skimage.color import label2rgb
import matplotlib.patches as patches
import cv2 as cv
import random
from PIL import Image
import PIL.ImageDraw as ImageDraw
import pandas as pd
import torch.optim as optim
from torch.autograd import Variable
import imageio
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from datetime import datetime
import json
from tune_functions import IMbags_single_slide,Attention_lenet,train,test_coarse,test_detail
import argparse

parser = argparse.ArgumentParser(description="Please specify the test patient ID")
parser.add_argument("test_patient", type=int, help = "Please specify the test patient ID")
parser.add_argument("mode", help = "Please specify uniform or multiscale mode")
parser.add_argument("architecture", help = "Please specify lenet or unet architecture")
args_io = parser.parse_args()

if args_io.mode == "uniform":
    if args_io.architecture == "lenet":
        shape_source = [100]
    elif args_io.architecture == "resnet":
        shape_source = [250]
    else:
        print("Specified architecture hasn't been implemented :)")
elif args_io.mode == "multiscale":
    if args_io.architecture == "lenet":
        shape_source = [100,500]
    elif args_io.architecture == "resnet":
        shape_source = [250,500]
    else:
        print("Specified architecture hasn't been implemented :)")
else:
    print("You can only choose uniform mode or multiscale mode")
    

class Args:
    def __init__(self):
        self.discard_patients = [6,25]
        self.reg = 10e-5
        self.patch_shape_sd = 92
        self.num_bag = 10
        self.length_bag = 400
        self.nfold = 5
args = Args()
root_dir = "/cis/home/zwang/digital-immune/Results/grid_search/test_on_"+str(args_io.test_patient)+'/'+args_io.mode+'_'+args_io.architecture
if not os.path.exists(root_dir):
    os.makedirs(root_dir)
  

    
df_meta = pd.read_csv("meta_anders.csv")
all_patients = [i for i in list(df_meta.subject_id) if i not in args.discard_patients]
test_patient = [args_io.test_patient]
train_patients = [i for i in all_patients if i not in test_patient]

test_slides = [list(df_meta.loc[df_meta.subject_id == i,'slide_id'])[0] for i in test_patient]
train_slides = [list(df_meta.loc[df_meta.subject_id == i,'slide_id'])[0] for i in train_patients]
print("test_patient",test_patient)
print("train_patients",train_patients)
print("test_slides",test_slides)
print("train_slides",train_slides)

def pack_bags(Patients,df_meta,args,shuffle):
    Slides = [list(df_meta.loc[df_meta.subject_id == i,'slide_id'])[0] for i in Patients]
    for i in range(len(Slides)):
        slide_ID = Slides[i]
        bag_set = IMbags_single_slide(slide_id = slide_ID,
                                          num_bag = args.num_bag,
                                          length_bag = args.length_bag,
                                          shape_source = shape_source,
                                          label = int(df_meta.loc[df_meta.slide_id == slide_ID,'label']),
                               transform = transforms.Compose([
                                          transforms.Resize(args.patch_shape_sd),
                                          transforms.ToTensor()
                                      ]))

        if i == 0:
            Bags_set = bag_set
        else:
            Bags_set = data_utils.ConcatDataset([Bags_set,bag_set])
    Bags_loader = data_utils.DataLoader(Bags_set, batch_size = 1, shuffle = shuffle)
    return Bags_loader

def split_patients(Patients,nfold):
    Patients_list = []
    fold_size = len(Patients)//nfold
    left_patients = Patients[-(len(Patients)%nfold):]
    for i in range(nfold):
        Patients_list.append(
           { "val":Patients[int(i*fold_size):int((i+1)*fold_size)]}
        )
    Patients_list[-1]['val'].extend(left_patients)
    for i in Patients_list:
        i['train'] = [j for j in Patients if j not in i['val']]
    return Patients_list

def cross_validation(Patients,df_meta,args,args_tune,epochs = [10,20,30,50]):
    Patients_list  = split_patients(Patients,args.nfold)
    for split in Patients_list:
        model=Attention_lenet()
        model.cuda()
        optimizer = optim.Adam(model.parameters(),lr=args_tune['initial_lr'], weight_decay =args.reg)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = args_tune['step'], gamma = args_tune['gamma'])
        split['Train_accuracy'] = []
        split['Val_accuracy'] = []
        split['Loss_train'] = []
        split['Loss_val']=[]
        val_loader = pack_bags(split['val'],df_meta,args,shuffle = False)
        start_time = datetime.now()
        for epoch in range(1,epochs[-1]+1):  
            train_loader = pack_bags(split['train'],df_meta,args,shuffle = True)
            train_accuracy, loss_train = train(epoch, train_loader, model, optimizer, scheduler)
            val_accuracy, loss_val = test_coarse(val_loader, model)
            print("\nepoch = {}, accuracy in training set is {}, loss in train set is {}".format(epoch, train_accuracy,  loss_train))
            print("epoch = {}, accuracy in validation set is {}, loss in valiation set is {}".format(epoch, val_accuracy, loss_val))
            split['Train_accuracy'].append(train_accuracy)
            split['Val_accuracy'] .append(val_accuracy)
            split['Loss_train'] .append(loss_train)
            split['Loss_val'].append(loss_val)
            scheduler.step()
            if epoch in epochs:
                train_loader = pack_bags(split['train'],df_meta,args,shuffle = False)
                train_accuracy, train_loss, (train_prob, train_labels), _ = test_detail(train_loader, model)
                val_accuracy, val_loss, (val_prob, val_labels), _ = test_detail(val_loader, model)
                train_labels =[i[0].item() for i in train_labels]
                val_labels =[i[0].item() for i in val_labels]
                train_prob =[i[0][0].item() for i in train_prob]
                val_prob =[i[0][0].item() for i in val_prob]
                #print(train_labels,train_prob)
                split['labels_train_'+str(epoch)] = train_labels
                split['prob_train_'+str(epoch)] = train_prob
                split['labels_val_'+str(epoch)] = val_labels
                split['prob_val_'+str(epoch)] = val_prob
                
                train_auc = roc_auc_score(train_labels, train_prob)
                fpr_train, tpr_train, _ = roc_curve(train_labels, train_prob)           
                split['fpr_train_'+str(epoch)] = fpr_train
                split['tpr_train_'+str(epoch)] = tpr_train
                split['auc_train_'+str(epoch)] = train_auc
                try:
                    val_auc = roc_auc_score(val_labels, val_prob)
                    fpr_val, tpr_val, _ = roc_curve(val_labels, val_prob)
                    split['fpr_val_'+str(epoch)] = fpr_val
                    split['tpr_val_'+str(epoch)] = tpr_val
                    split['auc_val_'+str(epoch)] = val_auc
                except:
                    split['fpr_val_'+str(epoch)] = None
                    split['tpr_val_'+str(epoch)] = None
                    split['auc_val_'+str(epoch)] = None
                end_time = datetime.now()
                split['time_'+str(epoch)] = (end_time-start_time).seconds/60
    return Patients_list

Args_tune = {
    'initial_lr':[10e-5,5*10e-6],
    'step':[10,20,5],
    'gamma':[0.5,0.75,1]
}

Version = 1
Grid_result = {}
for i in Args_tune['initial_lr']:
    for j in Args_tune['step']:
        for k in Args_tune['gamma']:
            if j!=10 and k==1:
                continue
            args_tune = {'initial_lr':i,'step':j,'gamma':k }
            Grid_result['initial_lr'] = i
            Grid_result['step'] = j
            Grid_result['gamma'] = k
            Grid_result['start_time'] = datetime.now()
            print(i,j,k)
            Patients_list = cross_validation(train_patients,df_meta,args,args_tune=args_tune,epochs = [1])
            for fold in range(len(Patients_list)):
                Grid_result['fold'+str(fold+1)] = Patients_list[fold]
            Grid_result['end_time'] = datetime.now()
            with open(root_dir+'/'+str(Version)+'.sav', 'wb') as handle:
                pickle.dump(Grid_result, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.close()
            Version+=1
