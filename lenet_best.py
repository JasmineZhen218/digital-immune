# -*- coding: utf-8 -*-
import openslide
import numpy as np
import os
import pickle
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
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
from tune_functions import IMbags_single_slide,Attention_lenet,train,test_coarse,test_detail,summary_cv,summarfy_grid_search
from sm_gs import *
import argparse
torch.backends.cudnn.benchmark=True

parser = argparse.ArgumentParser(description="Please specify the test patient ID")
parser.add_argument("test_patient", type=int, help = "Please specify the test patient ID")
parser.add_argument("mode", help = "Please specify uniform or multiscale mode")
parser.add_argument("architecture", help = "Please specify lenet or unet architecture")
parser.add_argument("gpu", help = "Choose a GPU")
args_io = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args_io.gpu

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
        self.patch_shape_sd = 92
        self.num_bag = 10
        self.length_bag = 400
        self.reg = 10e-5

args = Args()
root_dir = "/home/zhenzhen/ashel-slide/immnue/Fine_tunning/results/test_on_"+str(args_io.test_patient)+'/'+args_io.mode+'_'+args_io.architecture
if not os.path.exists(root_dir):
    os.makedirs(root_dir)  
    

df_meta = pd.read_csv("/home/zhenzhen/ashel-slide/meta_anders.csv")
all_patients = [i for i in list(df_meta.subject_id) if i not in args.discard_patients]
test_patient = [args_io.test_patient]
train_patients = [i for i in all_patients if i not in test_patient]

test_slides = [list(df_meta.loc[df_meta.subject_id == i,'slide_id'])[0] for i in test_patient]
train_slides = [list(df_meta.loc[df_meta.subject_id == i,'slide_id'])[0] for i in train_patients]
print("test_patient",test_patient)
print("train_patients",train_patients)
print("test_slides",test_slides)
print("train_slides",train_slides)

Best = {}
df, Validation_result, Training_result = summary_grid_search(root_dir)
index = np.where(Validation_result_arr == np.max(Validation_result_arr))
if index[1]==0:
    epoch = 10
elif index[1]==1:
    epoch = 20
elif index[1]==2:
    epoch = 50    

Version = index[0][0]+1
lr = df.loc[Version,'ilr']
step_size = df.loc[Version,'step']
gamma = df.loc[Version,'gamma']
print("Best version:", Version, epochs,lr,step_size,gamma)
Best['version'] = Version
Best['lr'] = lr
Best['step_size'] = step_size
Best['gamma'] = gamma
Best['epoch'] = epoch

model=Attention_lenet()
model.cuda()
optimizer = optim.Adam(model.parameters(),lr=lr, weight_decay =args.reg)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = step_size, gamma = gamma)
    

Train_accuracy = []
Loss_train = []
Epochs = np.arange(1,epochs+1).tolist()
for epoch in range(1,epochs+1):  
    for i in range(len(train_slides)):
        slide_ID = train_slides[i] 
        bag_train_set = IMbags_single_slide(slide_id = slide_ID,
                                          num_bag = args.num_bag,
                                          length_bag = args.length_bag,
                                          shape_source = shape_source,
                                          label = int(df_meta.loc[df_meta.slide_id == slide_ID,'label']),
                               transform = transforms.Compose([
                                          transforms.Resize(args.patch_shape_sd),
                                          transforms.ToTensor()
                                      ]))
        if i == 0:
            Bags_train_set = bag_train_set
        else:
            Bags_train_set = data_utils.ConcatDataset([Bags_train_set,bag_train_set])
    
    train_loader = data_utils.DataLoader(Bags_train_set, batch_size = 1, shuffle = True)
    train_accuracy, loss_train = train(epoch, train_loader, model, optimizer, scheduler)
    print("\nepoch = {}, accuracy in training set is {}, loss in train set is {}".format(epoch, train_accuracy,  loss_train))
    Train_accuracy.append(train_accuracy)
    Loss_train.append(loss_train)
    scheduler.step()
        
Best['train_accuracy'] = Train_accuracy
Best['train_loss'] = Train_loss
    
torch.save(model.state_dict(),root_dir+'best.pth')



# Test the model
df_meta = df_meta.sort_values(by = ['label','subject_id'] )
all_patients = [i for i in list(df_meta.subject_id) if i not in args.discard_patients]
train_patients = [i for i in all_patients if i not in test_patient]
test_slides = [list(df_meta.loc[df_meta.subject_id == i,'slide_id'])[0] for i in test_patient]
train_slides = [list(df_meta.loc[df_meta.subject_id == i,'slide_id'])[0] for i in train_patients]
print("test_patients",test_patient)
print("train_patients",train_patients)
print("test_slides",test_slides)
print("train_slides",train_slides)

for j in range(len(test_slides)):
    slide_ID = test_slides[j]
    bag_test_set = IMbags_single_slide(slide_id = slide_ID,
                                          num_bag = args.num_bag,
                                          length_bag = args.length_bag,
                                          shape_source = shape_source,
                                          label = int(df_meta.loc[df_meta.slide_id == slide_ID,'label']),
                               transform = transforms.Compose([
                                          transforms.Resize(args.patch_shape_sd),
                                          transforms.ToTensor()
                                      ]))
if j == 0:
            Bags_val_set = bag_val_set
        else:
            Bags_val_set = data_utils.ConcatDataset([Bags_val_set,bag_val_set])
    Bags_val_loader = data_utils.DataLoader(Bags_val_set, batch_size = 1, shuffle = False)
    val_loader = Bags_val_loader