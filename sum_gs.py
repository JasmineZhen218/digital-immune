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
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
from visualize import *
import matplotlib
import os
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
def sort_bag(labels, prob, num_bag_per_slide):
    label_slide = [labels[i] for i in range(len(labels)) if i%num_bag_per_slide==0]
    sort_index_slide = np.argsort(label_slide)
    for i in range(len(sort_index_slide)):

        slide_id = sort_index_slide[i]
        if i == 0:
            sort_index_bag = np.arange(slide_id*num_bag_per_slide,(slide_id+1)*num_bag_per_slide).tolist()
        else:
            sort_index_bag.extend(np.arange(slide_id*num_bag_per_slide,(slide_id+1)*num_bag_per_slide).tolist())
    labels_sorted = np.array(labels)[sort_index_bag].tolist()
    prob_sorted = np.array(prob)[sort_index_bag].tolist()
    return labels_sorted,prob_sorted

def visualize_one_cv(result,epoch = None, legend=False):
    f,ax = plt.subplots(1,5,figsize=(50,6))
    ax[0].plot(result['Train_accuracy'],color='darkgreen',label="Train accuracy")
    ax[0].plot(result['Val_accuracy'],color="orange",label='Validation accuracy')
    if legend:
        ax[0].legend(bbox_to_anchor=(-0.05, -0.35,1.1,0.2),
                  ncol=2, fancybox=True, shadow=True,mode="expand")
    ax[0].set(title="Accuracy", ylim=[0,1],xlabel="epoch")
    ax[0].axvline(x=10,color='k')
    ax[0].axvline(x=20,color='k')
    ax[0].axvline(x=50,color='k')

    ax[1].plot(result['Loss_train'],color="darkgreen",label="Train loss")
    ax[1].plot(result['Loss_val'],color="orange",label='Validation loss')
    if legend:
        ax[1].legend(bbox_to_anchor=(-0.05, -0.35,1.1,0.2),
                  ncol=2, fancybox=True, shadow=True,mode="expand")
    ax[1].set(title="Loss",xlabel="epoch")

    ax[1].axvline(x=10,color='k')
    ax[1].axvline(x=20,color='k')
    ax[1].axvline(x=50,color='k')

    if epoch is not None:
        ax[2].plot(result['fpr_train_'+str(epoch)],result['tpr_train_'+str(epoch)],color = 'cyan',label="T-AUC=%.2f"%(result['auc_train_'+str(epoch)]))
        fpr,tpr,_ = roc_curve(result['labels_val_'+str(epoch)],result['prob_val_'+str(epoch)])
        ax[2].plot(fpr,tpr,color='yellow',label="V-AUC=%.2f"%(roc_auc_score(result['labels_val_'+str(epoch)],result['prob_val_'+str(epoch)])))
    else:
        ax[2].plot(result['fpr_train_10'],result['tpr_train_10'],color = 'cyan',label="T-AUC=%.2f(E10)"%(result['auc_train_10']))
        ax[2].plot(result['fpr_train_20'],result['tpr_train_20'],color = 'springgreen',label="T-AUC=%.2f(E20)"%(result['auc_train_20']))
        ax[2].plot(result['fpr_train_50'],result['tpr_train_50'],color = 'darkgreen',label="T-AUC=%.2f(E50)"%(result['auc_train_50']))
        fpr,tpr,_ = roc_curve(result['labels_val_10'],result['prob_val_10'])
        ax[2].plot(fpr,tpr,color='yellow',label="V-AUC=%.2f(E10)"%(roc_auc_score(result['labels_val_10'],result['prob_val_10'])))
        fpr,tpr,_ = roc_curve(result['labels_val_20'],result['prob_val_20'])
        ax[2].plot(fpr,tpr,color='orange',label="V-AUC=%.2f(E20)"%(roc_auc_score(result['labels_val_10'],result['prob_val_20'])))
        fpr,tpr,_ = roc_curve(result['labels_val_50'],result['prob_val_50'])
        ax[2].plot(fpr,tpr,color='red',label="V-AUC=%.2f(E50)"%(roc_auc_score(result['labels_val_10'],result['prob_val_50'])))
   
    if legend:
        ax[2].legend(bbox_to_anchor=(-0.05, -0.35,1.1,0.2),
                      ncol=2, fancybox=True, shadow=True,mode="expand")
    ax[2].set(title="ROCs in training set",xlabel='False positive rate',ylabel="True positive rate")
    
    
    if epoch is not None:
        labels, prob = sort_bag(result['labels_train_'+str(epoch)], result['prob_train_'+str(epoch)], num_bag_per_slide=epoch)
        x=np.arange(len(labels)-np.sum(labels), len(labels))
        ax[3].plot(prob,color = 'cyan', label ='E'+str(epoch))
    else:
        labels, prob = sort_bag(result['labels_train_10'], result['prob_train_10'], num_bag_per_slide=10)
        x=np.arange(len(labels)-np.sum(labels), len(labels))
        ax[3].plot(prob,color = 'cyan', label ='E10')
        labels, prob = sort_bag(result['labels_train_20'], result['prob_train_20'], num_bag_per_slide=10)
        x=np.arange(len(labels)-np.sum(labels), len(labels))
        ax[3].plot(prob,color = 'springgreen',label ='E20')
        labels, prob = sort_bag(result['labels_train_50'], result['prob_train_50'], num_bag_per_slide=10)
        x=np.arange(len(labels)-np.sum(labels), len(labels))
        ax[3].fill_between(x=x,y1=[0]*len(x),y2=[1]*len(x),color='red',alpha=0.1,label = "Label=1")
        ax[3].plot(prob,color = 'darkgreen',label ='E50')
    for i in range(len(labels)//10):
        if i==0:
            continue
        position = i*10
        ax[3].axvline(x=position,color='k')
    if legend:
        ax[3].legend(bbox_to_anchor=(0, -0.3,1,0.1),
                  ncol=4, fancybox=True, shadow=True,mode='expand')
    ax[3].set(title="Predicted results in training set", xlabel='Bag ID')

    if epoch is not None:
        labels, prob = sort_bag(result['labels_val_'+str(epoch)], result['prob_val_'+str(epoch)], num_bag_per_slide=epoch)
        ax[4].plot(prob,color='yellow',label ='E'+str(epoch))    
    else:
        labels, prob = sort_bag(result['labels_val_10'], result['prob_val_10'], num_bag_per_slide=10)
        x=np.arange(len(labels)-np.sum(labels), len(labels))
        ax[4].plot(prob,color='yellow',label ='E10')    
        labels, prob = sort_bag(result['labels_val_20'], result['prob_val_20'], num_bag_per_slide=10)
        x=np.arange(len(labels)-np.sum(labels), len(labels))
        ax[4].plot(prob,color='orange',label ='E20')   
        labels, prob = sort_bag(result['labels_val_50'], result['prob_val_50'], num_bag_per_slide=10)
        x=np.arange(len(labels)-np.sum(labels), len(labels))
        ax[4].plot(prob,color='red',label ='E50')

    ax[4].fill_between(x=x,y1=[0]*len(x),y2=[1]*len(x),color='red',alpha=0.1,label = "Label=1")
    for i in range(len(labels)//10):
        if i==0:
            continue
        position = i*10
        ax[4].axvline(x=position,color='k')    
    if legend:
        ax[4].legend(bbox_to_anchor=(0, -0.3,1,0.1),
                  ncol=4, fancybox=True, shadow=True,mode='expand')
    ax[4].set(title="Predicted results in validation set", xlabel='Bag ID')
    
def summary_one_cv(Result,key,epoch):
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
    
    
def summary_grid_search(root_dir):
    df = pd.DataFrame()
    filenames = os.listdir(root_dir)
    Training_result = np.zeros((len(filenames),3))
    Validation_result = np.zeros((len(filenames),3))
    index = 0
    for Version in range(1,len(filenames)+1):
        with open(root_dir+'/'+str(Version)+'.sav', 'rb') as handle:
            Result = pickle.load(handle)

        df.loc[index,'version']=Version
        
        df.loc[index,'ilr']=Result['initial_lr']
        df.loc[index,'step']=Result['step']
        df.loc[index,'gamma']=Result['gamma']

        mean,std = summary_one_cv(Result,'auc_train',10)
        df.loc[index,'mean_auc_train_10']=mean
        df.loc[index,'std_auc_train_10']=std
        Training_result[index,0]=mean

        mean,std = summary_one_cv(Result,'auc_val',10)
        df.loc[index,'mean_auc_val_10']=mean
        df.loc[index,'std_auc_val_10']=std
        Validation_result[index,0]=mean

        mean,std = summary_one_cv(Result,'auc_train',20)
        df.loc[index,'mean_auc_train_20']=mean
        df.loc[index,'std_auc_train_20']=std
        Training_result[index,1]=mean

        mean,std = summary_one_cv(Result,'auc_val',20)
        df.loc[index,'mean_auc_val_20']=mean
        df.loc[index,'std_auc_val_20']=std
        Validation_result[index,1]=mean

        mean,std = summary_one_cv(Result,'auc_train',50)
        df.loc[index,'mean_auc_train_50']=mean
        df.loc[index,'std_auc_train_50']=std
        Training_result[index,2]=mean

        mean,std = summary_one_cv(Result,'auc_val',50)
        df.loc[index,'mean_auc_val_50']=mean
        df.loc[index,'std_auc_val_50']=std
        Validation_result[index,2]=mean
        
        index += 1
    return df, Validation_result, Training_result


def visulaize_grid_search(Validation_result,cmap=None):
    f,ax = plt.subplots(1,1,figsize=(10,5))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    Validation_result_show = np.zeros((Validation_result.shape[0],27))
    for i in range(Validation_result.shape[0]):
        for j in range(3):
            Validation_result_show[i,j*9: (j+1)*9] = Validation_result[i,j]
    if cmap is None:
        im = ax.imshow(Validation_result_show)
    else:
        im = ax.imshow(Validation_result_show,cmap=cmap)
        
    f.colorbar(im, cax=cax, orientation='vertical')
        
    xlabel = ['10','20','50' ]
    x = [5,13,22]
    ax.set_xticks(x)
    ax.set_xticklabels(xlabel)
    ax.set_title("Visualization of Validation AUCs grid search")
    ax.set_ylabel("Version of parameter combination")
    ax.set_xlabel("Epochs")

    Version_best, epoch_id_best = np.where(Validation_result==np.max(Validation_result))
    rec = patches.Rectangle((epoch_id_best*9-0.5,Version_best-0.6),9,1.5,fill=False,edgecolor="r")
    ax.add_patch(rec)
    ax.text(epoch_id_best*9-0.5,Version_best-0.7, s="AUC = {:.4F}".format(np.max(Validation_result)))

def visualize_best_results(root_dir,Validation_result_arr,df,legend=False):
    index = np.where(Validation_result_arr == np.max(Validation_result_arr))
    if index[1]==0:
        epoch = 10
    elif index[1]==1:
        epoch = 20
    elif index[1]==2:
        epoch = 50    

    Version = index[0][0]+1
    print(Version)
    with open(root_dir+'/'+str(Version)+'.sav', 'rb') as handle:
        Result = pickle.load(handle)
    
    print("Version = {}, LR = {}, Step = {}, Gamma = {}".format( Version,
                                                                list(df.loc[df.version==Version,'ilr'])[0],
                                                                list(df.loc[df.version==Version,'step'])[0],
                                                                list(df.loc[df.version==Version,'gamma'])[0] ))
    
    f,ax = plt.subplots(1,2,figsize=(12,5))
    for fold in range(1,6):
        fpr,tpr,_ = roc_curve(Result['fold'+str(fold)]['labels_train_'+str(epoch)],Result['fold'+str(fold)]['prob_train_'+str(epoch)])
        ax[0].plot(fpr,tpr,label="AUC=%.2f"%(roc_auc_score(Result['fold'+str(fold)]['labels_train_'+str(epoch)],Result['fold'+str(fold)]['prob_train_'+str(epoch)])))
    if legend:
        ax[0].legend()
    mean,std = summary_one_cv(Result,'auc_train',epoch)
    ax[0].set(title="Training(%d epoch), AUC= %.2f $\pm$ %.2f"%(epoch,mean,std))
    train_mean , train_std = mean,std
    for fold in range(1,6):
        fpr,tpr,_ = roc_curve(Result['fold'+str(fold)]['labels_val_'+str(epoch)],Result['fold'+str(fold)]['prob_val_'+str(epoch)])
        ax[1].plot(fpr,tpr,label="AUC=%.2f"%(roc_auc_score(Result['fold'+str(fold)]['labels_val_'+str(epoch)],Result['fold'+str(fold)]['prob_val_'+str(epoch)])))
    if legend:
        ax[1].legend()
    mean,std = summary_one_cv(Result,'auc_val',epoch)
    ax[1].set(title="Validation(%d epoch), AUC= %.2f $\pm$ %.2f"%(epoch,mean,std))
    test_mean, test_std = mean,std
    return train_mean, train_std, test_mean, test_std