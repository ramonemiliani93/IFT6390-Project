import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


from sklearn.decomposition import PCA

import argparse
import logging
import os

import numpy as np
import torch


import utils
from model.net import LinearRegression, MLP, CNN
import model.data_loader as data_loader

mnist_classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
              '#bcbd22', '#17becf']

parser = argparse.ArgumentParser()
parser.add_argument('model', default=None, choices=['linear', 'mlp', 'cnn'], help="Model to train")
parser.add_argument('dataset', default=None, choices=['fashion', 'cifar'], help="Model to train")
parser.add_argument('checkpoint', default=None, help='Model weights')
parser.add_argument('--data_dir', default='data', help="Directory that will contain dataset")
parser.add_argument('--model_dir', default='experiments/base_model', help="Directory containing params.json")
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")  # 'best' or 'train'


def plot_embeddings_3D(embeddings, targets):

    pca = PCA(n_components=3)
    pca.fit(embeddings)
    x = pca.transform(embeddings)

    fig = plt.figure(1, figsize=(8, 6))
    plt.clf()
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

    # for i in np.unique(targets):
    ax.scatter(x[:, 0],
               x[:, 1],
               x[:, 2], c=targets, cmap=plt.cm.nipy_spectral, edgecolor='k')

    plt.show()


def plot_embeddings_2D(embeddings, targets):

    x = PCA(n_components=2).fit_transform(embeddings)

    plt.figure(1, figsize=(8, 6))
    plt.clf()

    # for i in np.unique(targets):
    plt.scatter(x[:, 0], x[:, 1], c=targets, cmap=plt.cm.nipy_spectral, edgecolor='k')

    plt.show()


def extract_embeddings(dataloader, model):
    with torch.no_grad():
        model.eval()
        embeddings, labels = [], []
        for images, target in dataloader:
            if torch.cuda.is_available():
                images = images.cuda()
            embeddings.append(model.extract_features(images).data.cpu().numpy())
            labels.append(target.numpy())
        embeddings = np.concatenate(embeddings)
        labels = np.concatenate(labels)
    return embeddings, labels


if __name__ == '__main__':
    if __name__ == '__main__':

        # Load the parameters from json file
        args = parser.parse_args()
        json_path = os.path.join(args.model_dir, 'params.json')
        assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
        params = utils.Hyperparameters(json_path)

        # use GPU if available
        params.cuda = torch.cuda.is_available()

        # Set the random seed for reproducible experiments
        torch.manual_seed(230)
        if params.cuda:
            torch.cuda.manual_seed(230)

        # Set the logger
        utils.set_logger(os.path.join(args.model_dir, 'lg.log'))

        # Create the input data pipeline
        logging.info("Loading the datasets...")

        # fetch dataloaders
        train_dl, val_dl = data_loader.fetch_train_dataloaders(args.dataset, args.data_dir, params)

        logging.info("- done.")

        # Define the model and optimizer
        choices = {
            'linear': LinearRegression().cuda() if params.cuda else LinearRegression(),
            'mlp': MLP().cuda() if params.cuda else MLP(),
            'cnn': CNN().cuda() if params.cuda else CNN()
        }
        model = choices[args.model]
        utils.load_checkpoint(args.checkpoint, model)

        train_embeddings, train_labels = extract_embeddings(train_dl, model)
        val_embeddings, val_labels = extract_embeddings(val_dl, model)

        plot_embeddings_3D(train_embeddings, train_labels)