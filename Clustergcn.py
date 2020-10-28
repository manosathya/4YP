#!/usr/bin/python
import numpy as np
import pandas as pd
import torch

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix 

import copy
from os.path import join

from data import load
import gcn_config as cfg
from functions import mkdir_p

import gc

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
save_train = cfg.save_train

def train(model, data, optimizer, criterion):

    """ Training the Model """
    v_acc, v_loss = 0,0
#    predictions, y_true = [],[]
    
    dataloaders = data[0]
######################################################################################################3333    
    #Training and val phase in each epoch      
    for phase in ['train', 'val']: 
        if phase == 'train':
            model.train() 
            length = data[1]['t_size']
        else:
            model.eval()  
            length = data[1]['v_size']
            
        running_loss = 0.0
        running_corrects = 0 
        
        #Iterating over data
        for cluster in dataloaders[phase]:
            
            cluster = cluster.to(device)
            _, targets = cluster.y.max(dim=1)
            
            optimizer.zero_grad()
            
            with torch.set_grad_enabled(phase == 'train'):            
                outputs = model(cluster)
                loss = criterion(outputs, targets)            
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
        

            _, predicted = torch.max(outputs.data,1)

            running_loss += loss.item() * cluster.x.size(0)
            running_corrects += torch.sum(predicted == targets.data) 
            
      
       #     if phase == "val":
        #        predicted, targets = predicted.to("cpu"), targets.to("cpu")
         #       predictions = np.append(predictions, predicted.numpy(), axis=None)
          #      y_true = np.append(y_true, targets.numpy(), axis=None)
        
        epoch_loss = running_loss / length
        epoch_acc = running_corrects.double() / length

        # deep copy the model
        if phase == 'val': 
           # s = metrics(y_true,predictions,average='macro')
            v_acc = epoch_acc
            v_loss = epoch_loss
    gc.collect()        
########################################################################################################################33           
    return v_acc, v_loss

def plt_train(hist):             
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(18, 6))
    for i in [0,1]:
        graph = ["Accuracy", hist[0], hist[1]] if i == 0 else ["Loss", hist[2], hist[3]]
        
        fig.suptitle(Net_Name.upper())
        ax[i].set(title= graph[0] + " vs. Number of Training Epochs", 
                  xlabel = "Training Epochs", ylabel= graph[0], 
                  xticks = np.arange(0, epoch+1, 5))
        ax[i].plot(range(1,epoch + 2), graph[1], label="Training")
        ax[i].plot(range(1,epoch + 2), graph[2], label="Validation")
        ax[i].axvline(x= best_epoch + 1, c='r', linewidth=1, ls='dashed')
        ax[i].legend()
        
    if save_train.upper() == "Y":
        plt.savefig(join("Results","Training_Curves","GCN", Net_Name + "_train.png")) 
        
      
def test(adj, best_model):

    test_loader, test_size = load(adj,"test")
    correct, predictions, y_true = 0,[],[]
    model = copy.deepcopy(best_model)
    with torch.no_grad():
        for cluster in test_loader:
            
            cluster = cluster.to(device)
            _, targets = cluster.y.max(dim=1)
                
            outputs = model(cluster)
            
            _, predicted = torch.max(outputs.data, 1)
            
            
            correct += torch.sum(predicted == targets.data) 
            predicted = predicted.to("cpu")
            predictions = np.append(predictions, predicted.numpy(), axis=None)
            
            targets = targets.to("cpu")
            y_true = np.append(y_true, targets.numpy(), axis=None)
    accuracy = (100 * correct / test_size).cpu().numpy()         
    print("Accuracy of the network on the test data %d %%" %(accuracy))   
    
    s = list(metrics(y_true,predictions,average='macro'))
    scores = ['%.3f' % i for i in s[:-1]]
    del s
    scores.append(accuracy)
    print(scores)  
    return scores, y_true, predictions
 
def plt_cm(y_true, predictions):    
    """Plotting confusion matrices"""
    headers = ['track_id', 'genre', 'album_id', 'set', 'artist_id']
    Dataset = pd.read_csv(join("splits", "MSD-I_dataset.tsv"), header = 0, names = headers + [''] , sep = '\t', usecols = headers)
    #Convert from one hot encoded to column vector
    cm = confusion_matrix(y_true, predictions)
    classes = sorted(Dataset.genre.unique())
    
    fig, axs = plt.subplots(nrows=1, ncols=2,figsize=(15, 8))
    fig.subplots_adjust(hspace = 0, wspace=0.5)
    fig.suptitle(Net_Name.upper(), x = 0.1, y = 0.93, size = 15)
    i = 0
    for ax in axs:  
        ax.set_title("W/o Normalisation")
        if i == 1:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            ax.set_title("With Normalisation")
            
        im = ax.imshow(cm, cmap='Blues')
        ax.figure.colorbar(im, ax=ax, shrink = 0.7)
        # We want to show all ticks...
        ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]), 
               yticklabels = classes, xlabel='Predicted label', ylabel='True label')
        ax.set_xticklabels(classes, rotation= 90)
        i+=1

    fig.tight_layout()
    
    if cfg.save_test.upper() == "Y":
        mkdir_p(join("Results","CMs","GCN"))
        fig.savefig(join("Results","CMs","GCN",Net_Name + ".png"))    
    i +=1
