import argparse
import logging
import os
import json
import sys
from subprocess import check_call
import pandas as pd
from synthesize_results import get_best_metrics, aggregate_metrics
from matplotlib import pyplot as plt
import torch
import utils
import model.data_loader as data_loader
from model.net import LinearRegression, MLP, CNN
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--parent_dir', default='experiments/grid_search/results',
                    help='Directory containing results of experiments')
parser.add_argument('--best_dir', default='experiments/best_models')
parser.add_argument('--bottleneck_dir', default='experiments/bottlenecks')

PYTHON = sys.executable

mnist_classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
          '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
          '#bcbd22', '#17becf']

figures_dir = 'bottleneck_figures'


def extract_embeddings(dataloader, model):
    with torch.no_grad():
        model.eval()
        embeddings, labels = [], []
        for images, target in dataloader:
            if torch.cuda.is_available():
                images = images.cuda()
            embeddings.append(model.extract_bottleneck(images).data.cpu().numpy())
            labels.append(target.numpy())
        embeddings = np.concatenate(embeddings)
        labels = np.concatenate(labels)
    return embeddings, labels


def plot_embeddings_2D(embeddings, targets, name, bottleneck_dir):
    plt.figure(1, figsize=(8, 6))
    plt.clf()

    # for i in np.unique(targets):
    plt.scatter(embeddings[:, 0], embeddings[:, 1], c=targets, cmap=plt.cm.nipy_spectral, edgecolor='k')
    plt.tight_layout()
    plt.savefig(os.path.join(bottleneck_dir, name + '.jpg'))
    plt.show()


if __name__ == '__main__':
    args = parser.parse_args()
    results = pd.DataFrame(columns=['model', 'dataset', 'loss_fn', 'lr', 'bs', 'epochs', 'l2', 'l1', 'acc', 'loss'])
    # Aggregate metrics from args.parent_dir directory
    metrics = dict()
    results = aggregate_metrics(args.parent_dir, metrics)
    results[['l1', 'l2']] = results[['l1', 'l2']].apply(pd.to_numeric)
    best_models = pd.DataFrame(columns=['model', 'dataset', 'loss_fn', 'lr', 'bs', 'epochs', 'l2', 'l1', 'acc', 'loss'])
    best_models = get_best_metrics(args, best_models, results)

    best_models[['lr', 'bs', 'acc', 'loss', 'epochs', 'acc_test', 'loss_test']] = best_models[
        ['lr', 'bs', 'acc', 'loss', 'epochs', 'acc_test', 'loss_test']].apply(
        pd.to_numeric)
    best_models_dir = os.path.join(args.parent_dir, 'best_models')
    bottleneck_dir = args.bottleneck_dir

    for index, row in best_models.iterrows():

        subdir = 'model__' + row['model'] + \
                 '___dataset__' + row['dataset'] + \
                 '___loss__' + row['loss_fn'] + \
                 '___learning_rate__' + str(row['lr']) + \
                 '___batch_size__' + str(row['bs']) + \
                 '___num_epochs__' + str(row['epochs']) + \
                 '___weight_decay__' + str(row['l2']) + \
                 '___l1_reg__' + str(row['l1'])

        subdir = subdir.replace('__0.0___', '__0___')
        if subdir[-3:] == '0.0':
            subdir = subdir.replace('___l1_reg__0.0', '___l1_reg__0')

        if not os.path.isdir(bottleneck_dir):
            os.mkdir(bottleneck_dir)
        json_params = {'learning_rate': row['lr'],
                       'batch_size': row['bs'],
                       'num_epochs': row['epochs'],
                       'weight_decay': row['l2'],
                       'l1_reg': row['l1']}

        exp_dir = os.path.join(bottleneck_dir, subdir)
        if not os.path.isdir(exp_dir):
            os.mkdir(exp_dir)
            with open(os.path.join(exp_dir, 'params.json'), 'w') as outfile:
                json.dump(json_params, outfile)

            if os.path.isfile(os.path.join(figures_dir, subdir + '.jpg')):
                continue
            cmd = "{python} train.py {} {} {} --bottleneck --model_dir={model_dir} --data_dir {data_dir}".format(
                row['model'],
                row['dataset'],
                row['loss_fn'],
                python=PYTHON,
                model_dir=exp_dir,
                data_dir='data')

            print(cmd)
            check_call(cmd, shell=True)

            params = utils.Hyperparameters(os.path.join(exp_dir, 'params.json'))
            params.cuda = torch.cuda.is_available()
            torch.manual_seed(230)
            if params.cuda:
                torch.cuda.manual_seed(230)
            utils.set_logger(os.path.join(exp_dir, 'lg.log'))
            logging.info("Loading the datasets...")

            # fetch dataloaders
            test_dl = data_loader.fetch_test_dataloader(row['dataset'], 'data', params)

            logging.info("- done.")
            choices = {
                'linear': LinearRegression().cuda() if params.cuda else LinearRegression(bottleneck=True),
                'mlp': MLP(bottleneck=True).cuda() if params.cuda else MLP(bottleneck=True),
                'cnn': CNN(bottleneck=True).cuda() if params.cuda else CNN(bottleneck=True)
            }
            model = choices[row['model']]
            utils.load_checkpoint(os.path.join(exp_dir, 'best.pth.tar'), model)

            test_embeddings, test_labels = extract_embeddings(test_dl, model)

            if not os.path.isdir(figures_dir):
                os.mkdir(figures_dir)

            string = row['model'] + '_' + row['dataset'] + '_' + row['loss_fn']
            plot_embeddings_2D(test_embeddings, test_labels, string, figures_dir)


