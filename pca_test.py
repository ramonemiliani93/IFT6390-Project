import sys
import os
import utils 
import torch
import model.net as net
import numpy as np
import matplotlib.pyplot as plt
import model.data_loader as data_loader
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import time
import os.path
# find . -name "best.pth.tar" >> directories.txt
filename = "directories.txt"


def get_model_files(file_source):
    file_names = open(file_source, "r").readlines()
    file_names = [name.replace("\n","") for name in file_names]
    line_models = [name for name in file_names if "__linear__" in name]
    mlp_models = [name for name in file_names if "__mlp__" in name]
    cnn_models = [name for name in file_names if "__cnn__" in name]
    return line_models,mlp_models,cnn_models



def load_model(my_model):
    empty_model = None
    if("linear" in my_model):
        empty_model = net.LinearRegression()
    elif("mlp" in my_model):
        empty_model = net.MLP()
    elif("cnn" in my_model):
        empty_model = net.CNN()
    loaded_model = utils.load_model(my_model,empty_model)
    return loaded_model


def plot_pca(tensor):
    """ Plots the PCA of the given tensor
    Args:
        tensor: the X values of the PCA
    """
    values = tensor.numpy()
    pca = PCA(2)
    transformed = pca.fit_transform(values)
    plt.scatter(transformed[:,0],transformed[:,1])
    for direction in pca.components_:
        new_d = pca.transform([direction])[0]
        plt.arrow(0,0,new_d[0],new_d[1])
    plt.show()

class param:
    def __init__(self,cuda,batch):
        self.batch_size = batch
        self.cuda = cuda
        return

def plot_pca_classes(features,classes,model_base,output):
    """ Plots (to file specified by model_base and output) the PCA of the given features
    Args:
        features: the X values of the PCA
        classes: the labels for each x in X
        model_base: the directory of the model used
        output: the name of the file output within the directory
    """
    pca = PCA(2)
    transformed = pca.fit_transform(features)
    fig = plt.figure(figsize=(10,6))
    labels = [str(classes[i]) for i in range(classes.shape[0])]
    plt.scatter(transformed[:,0],transformed[:,1],c=classes,cmap='nipy_spectral',label=labels)
    kind = None
    if("linear" in model_base):
        kind = "Linear"
    elif("mlp" in model_base):
        kind = "MLP"
    elif("cnn" in model_base):
        kind = "CNN"
    dataset = None
    if("fashion" in model_base):
        dataset = "Fashion"
    else:
        dataset = "CIFAR"
    plt.title("{}: {}".format(kind,dataset))
    file_path = model_base.replace(model_base.split("/")[-1],output)
    fig.savefig(file_path)
    # for direction in pca.components_:
    #     new_d = pca.transform([direction])[0]#/=np.linalg.norm(direction)
    #     plt.arrow(0,0,new_d[0],new_d[1])
    # plt.show()


def compute_features(model):
    """ Gets the values stored in the feature variables of the passed model
    Args:
        model: the model defined in net.py to get the features from
    """
    feats,labels=[],[]
    for i_batch, sample_batched in enumerate(val_dl):
        feats.append(loaded_model.extract_features(sample_batched[0]).detach().numpy())
        labels.append(sample_batched[1].numpy())
    feats = np.concatenate(feats)
    labels = np.concatenate(labels)
    return feats,labels

def compute_outputs(model):
    """ Gets the values stored in the feature variables of the passed model
    Args:
        model: the model defined in net.py to get the output from
    """
    feats,labels=[],[]
    for i_batch, sample_batched in enumerate(val_dl):
        feats.append(loaded_model.forward(sample_batched[0]).detach().numpy())
        labels.append(sample_batched[1].numpy())
    feats = np.concatenate(feats)
    labels = np.concatenate(labels)
    return feats,labels

linear, mlp, cnn = get_model_files(filename)
params = param(False,64)

train_dl, val_dl = data_loader.fetch_train_dataloaders('cifar', 'data', params)

for model in linear:
    loaded_model = load_model(model)
    if(not os.path.isfile(model.replace(model.split("/")[-1],"features_pca.png"))):
        feats,labels = compute_features(loaded_model)
        plot_pca_classes(feats,labels,model,"features_pca")
        del(feats,labels)
    if(not os.path.isfile(model.replace(model.split("/")[-1],"outputs_pca.png"))):
        feats,labels = compute_outputs(loaded_model)
        plot_pca_classes(feats,labels,model,"outputs_pca")

for model in mlp:
    loaded_model = load_model(model)
    if(not os.path.isfile(model.replace(model.split("/")[-1],"features_pca.png"))):
        feats,labels = compute_features(loaded_model)
        plot_pca_classes(feats,labels,model,"features_pca")
        del(feats,labels)
    if(not os.path.isfile(model.replace(model.split("/")[-1],"outputs_pca.png"))):
        feats,labels = compute_outputs(loaded_model)
        plot_pca_classes(feats,labels,model,"outputs_pca")

for model in cnn:
    loaded_model = load_model(model)
    if(not os.path.isfile(model.replace(model.split("/")[-1],"features_pca.png"))):
        feats,labels = compute_features(loaded_model)
        plot_pca_classes(feats,labels,model,"features_pca")
        del(feats,labels)
    if(not os.path.isfile(model.replace(model.split("/")[-1],"outputs_pca.png"))):
        feats,labels = compute_outputs(loaded_model)
        plot_pca_classes(feats,labels,model,"outputs_pca")
