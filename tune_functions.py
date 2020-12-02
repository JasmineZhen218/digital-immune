# -*- coding: utf-8 -*-
import openslide
import numpy as np
import os
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
import pickle


class IMbags_single_slide(data_utils.Dataset):
    def __init__(self, slide_id, length_bag, num_bag, shape_source, label,transform = None):
        self.slide_id = slide_id  
        self.label = label
        self.shape_source = shape_source
        self.num_bag = num_bag
        self.length_bag = length_bag
        self.transform = transform
        
        self.files = self.get_full_names()
        #print(len(self.files))
        self.bag_list = self.get_bag_list()
    
    def get_full_names(self):
        files = []
        for shape in self.shape_source: 
            for i in os.listdir('/cis/home/zwang/digital-immune/anders_immune/'+str(shape)+'/'+str(self.slide_id)):
                files.append('/cis/home/zwang/digital-immune/anders_immune/'+str(shape)+'/'+str(self.slide_id)+'/'+i)
        return files
        
    def get_bag_list(self):
        bag_list = []
        for i in range(self.num_bag):
            files = random.sample(self.files,k=self.length_bag)
            bag_list.append(files)
        return bag_list
    
   
    def __len__(self):
        return len(self.bag_list)
    
    def __getitem__(self,idx):
        
        patches_in_one_bag = []
        for file in self.bag_list[idx]:
            patch = Image.open(file)
            if self.transform is not None:
                patch=self.transform(patch)
            patches_in_one_bag.append(patch)
        Bag = np.stack(patches_in_one_bag,axis=0)
        return Bag,self.label
    
   
    def __len__(self):
        return len(self.bag_list)
    
    def __getitem__(self,idx):
        
        patches_in_one_bag = []
        for file in self.bag_list[idx]:
            patch = Image.open(file)
            if self.transform is not None:
                patch=self.transform(patch)
            patches_in_one_bag.append(patch)
        Bag = np.stack(patches_in_one_bag,axis=0)
        return Bag,self.label

class Attention_lenet(nn.Module):
    def __init__(self):
        super(Attention_lenet,self).__init__()
        self.L = 500
        self.D = 128
        self.K = 1
        
        self.dropout = nn.Dropout(p=0.5)

        self.feature_extractor_part1 = nn.Sequential(
            nn.Conv2d(3,20,kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2,stride=2),
            nn.Conv2d(20,50,kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(50,60,kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )
        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(60 * 8* 8, self.L),
            nn.ReLU()
        )
        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.L * self.K, 1),
            nn.Sigmoid()
        )
    def forward(self,x):
        x = x .squeeze(0)
        H = self.feature_extractor_part1(x)
        #print(H.shape)
        #H = H.view(-1, 60 * 8 * 8)
        H = torch.flatten(H, start_dim =1)
        #print(H.shape)
        #H = self.dropout(H)
        #print(H.shape)
        H = self.feature_extractor_part2(H)

        A = self.attention(H)
        A = torch.transpose(A,1,0)
        A = F.softmax(A,dim=1)

        M = torch.mm(A,H)
        M = self.dropout(M)

        Y_prob = self.classifier(M)
        Y_hat = torch.ge(Y_prob,0.5).float()
        #print(Y_prob.shape, Y_hat.shape)
        return Y_prob, Y_hat, A

    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        Y_prob, Y_hat,_ = self.forward(X)
        #print(Y, Y_hat, Y_hat.eq(Y))
        error = 1. - Y_hat.eq(Y).cpu().float().mean().data.item()

        return error, Y_hat, Y_prob
  
    def calculate_objective(self, X, Y):
        Y = Y.float()
        Y_prob, _, A = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood =-1. * (Y * torch.log(Y_prob)+(1. - Y) * torch.log(1. - Y_prob))

        return neg_log_likelihood, A

def train(epoch, loader, model, optimizer, scheduler):
    model.train()
    train_loss = 0.
    train_error = 0.
    optimizer.zero_grad()
    for batch_idx, (data,label) in enumerate(loader):
        #print(batch_idx,data.shape)
        bag_label = label
        data, bag_label = data.cuda(), bag_label.cuda()
        data, bag_label = Variable(data), Variable(bag_label)
        loss, _ = model.calculate_objective(data, bag_label)
        error, _, _ =model.calculate_classification_error(data, bag_label)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_loss += loss
        train_error += error
        del data
        del bag_label
        del loss
        del error
    train_loss /= len(loader)
    train_error /= len(loader)
    #print('Epoch: {}, Loss: {:.4f}, Train error : {:.4f}'.format(epoch, train_loss.cpu().numpy()[0], train_error))
    return 1 - train_error, train_loss.cpu().data.numpy()[0][0]
def test_coarse(loader, model):
    model.eval()
    test_loss = 0.
    test_error = 0.
  
    for batch_idx, (data, label) in enumerate(loader):
        bag_label = label
        data, bag_label = data.cuda(), bag_label.cuda()
        data, bag_label = Variable(data), Variable(bag_label)
    
        loss, _ = model.calculate_objective(data, bag_label)
        error, _, _ = model.calculate_classification_error(data, bag_label)
        
        test_loss += loss.cpu().data.numpy()[0]
        test_error += error
        del data
        del bag_label
    
    test_error /= len(loader)
    test_loss /= len(loader)

    return 1 - test_error, test_loss[0]


def test_detail(loader, model):
    model.eval()
    test_loss = 0.
    test_error = 0.
    Y_probs = []
    Labels = []
    Attention_weights_positive_bags = []
    Attention_weights_negative_bags = []
    Attention_weights_positive_hat_bags = []
    Attention_weights_negative_hat_bags = []
    for batch_idx, (data, label) in enumerate(loader):
        bag_label = label
        data, bag_label = data.cuda(), bag_label.cuda()
        data, bag_label = Variable(data), Variable(bag_label)
        loss, attention_weights = model.calculate_objective(data, bag_label)
        test_loss += loss.cpu().data.numpy()[0]
        error, predicted_label, Y_prob = model.calculate_classification_error(data, bag_label)
        test_error += error
        Y_probs.append(Y_prob.cpu().data.numpy())
        Labels.append(label)
        if label.numpy()[0]==1:
            Attention_weights_positive_bags.extend(attention_weights.cpu().data.numpy()[0].tolist())
        else:
            Attention_weights_negative_bags.extend(attention_weights.cpu().data.numpy()[0].tolist())
            
        if predicted_label.cpu().numpy()==1:
            Attention_weights_positive_hat_bags.extend(attention_weights.cpu().data.numpy()[0].tolist())
        else:
            Attention_weights_negative_hat_bags.extend(attention_weights.cpu().data.numpy()[0].tolist())
        del data
    test_error /= len(loader)
    test_loss /= len(loader)
    
    accuracy = 1 - test_error
    bag_level = (Y_probs, Labels)
    instance_level = {
        'actual positive':Attention_weights_positive_bags,
        'actual negative':Attention_weights_negative_bags,
        'predicted positive':Attention_weights_positive_hat_bags,
        'predicted negative':Attention_weights_negative_hat_bags,
        
    }
    return accuracy, test_loss, bag_level, instance_level        
        
def summary_cv(Result,key,epoch):
    if key == 'Train_accuracy' or key == 'Loss_val' or key == 'Loss_train' or key == 'Val_accuracy':
        summary = []
        for i in range(1,6):
            summary.append(Result['fold'+str(i)][key][epoch-1])
        return np.mean(summary),np.std(summary)
    else:
        summary = []
        for i in range(1,6):
            summary.append(Result['fold'+str(i)][key+'_'+str(epoch)])
        return np.mean(summary),np.std(summary)

    
def summarfy_grid_search(root_dir, Version_s, Version_e,num):
    df = pd.DataFrame()
    Validation_result = np.zeros((num,3))
    Training_result = np.zeros((num,3))
    for Version in range(Version_s,Version_e+1):
        with open(root_dir+'/'+str(Version)+'.sav', 'rb') as handle:
            Result = pickle.load(handle)
    
        df.loc[Version,'ilr']=Result['initial_lr']
        df.loc[Version,'step']=Result['step']
        df.loc[Version,'gamma']=Result['gamma']

        mean,std = summary_cv(Result,'auc_train',10)
        df.loc[Version,'mean_auc_train_10']=mean
        df.loc[Version,'std_auc_train_10']=std
        Training_result[Version-Version_s,0]=mean

        mean,std = summary_cv(Result,'auc_val',10)
        df.loc[Version,'mean_auc_val_10']=mean
        df.loc[Version,'std_auc_val_10']=std
        Validation_result[Version-Version_s,0]=mean

        mean,std = summary_cv(Result,'auc_train',20)
        df.loc[Version,'mean_auc_train_20']=mean
        df.loc[Version,'std_auc_train_20']=std
        Training_result[Version-Version_s,1]=mean

        mean,std = summary_cv(Result,'auc_val',20)
        df.loc[Version,'mean_auc_val_20']=mean
        df.loc[Version,'std_auc_val_20']=std
        Validation_result[Version-Version_s,1]=mean

        mean,std = summary_cv(Result,'auc_train',50)
        df.loc[Version,'mean_auc_train_50']=mean
        df.loc[Version,'std_auc_train_50']=std
        Training_result[Version-Version_s,2]=mean

        mean,std = summary_cv(Result,'auc_val',50)
        df.loc[Version,'mean_auc_val_50']=mean
        df.loc[Version,'std_auc_val_50']=std
        Validation_result[Version-Version_s,2]=mean
    return df, Validation_result, Training_result