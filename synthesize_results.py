"""Aggregates results from the metrics_eval_best_weights.json in a parent folder"""

import argparse
import json
import os
import pandas as pd
from tabulate import tabulate
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import shutil
from utils import safe_log

plt.style.use('seaborn-white')
plt.rcParams['font.monospace'] = 'Ubuntu Mono'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.titlesize'] = 20

parser = argparse.ArgumentParser()
parser.add_argument('--parent_dir', default='experiments/grid_search/results',
                    help='Directory containing results of experiments')
parser.add_argument('--best_dir', default='experiments/best_models')


def aggregate_metrics(parent_dir, metrics):
    """Aggregate the metrics of all experiments in folder `parent_dir`.
    Assumes that `parent_dir` contains multiple experiments, with their results stored in
    `parent_dir/subdir/metrics_dev.json`
    Args:
        parent_dir: (string) path to directory containing experiments results
        metrics: (dict) subdir -> {'accuracy': ..., ...}
    """
    # Get the metrics for the folder if it has results from an experiment
    results = pd.DataFrame(columns=['model', 'dataset', 'loss_fn', 'lr', 'bs', 'epochs', 'l1', 'l2', 'acc', 'loss'])

    for subdir in os.listdir(parent_dir):
        metrics_file = os.path.join(parent_dir, subdir, 'metrics_val_best_weights.json')
        if os.path.isfile(metrics_file):
            settings = subdir.split('___')
            subsettings = [subset.split('__') for subset in settings]
            pd_settings = [item[1] for item in subsettings]
            with open(metrics_file, 'r') as f:
                metrics[subdir] = json.load(f)
                pd_settings.append(metrics[subdir]['accuracy'])
                pd_settings.append(metrics[subdir]['loss'])
                pd_settings = np.asarray(pd_settings)
                row = pd.DataFrame(np.expand_dims(pd_settings, 0),
                                   columns=['model', 'dataset', 'loss_fn', 'lr',
                                            'bs', 'epochs', 'l1', 'l2', 'acc', 'loss'])
                results = results.append(row)

    return results


def metrics_to_table(metrics):
    # Get the headers from the first subdir. Assumes everything has the same metrics
    headers = metrics[list(metrics.keys())[0]].keys()
    table = [[subdir] + [values[h] for h in headers] for subdir, values in metrics.items()]
    res = tabulate(table, headers, tablefmt='pipe')

    return res


def generate_accuracy_plot(df, dataset, loss, model):
    # Generate 3D plots where color indicates accuracy
    acccuracy = [float(i) for i in df['acc'].tolist()]
    l1 = [safe_log(float(i)) for i in df['l1'].tolist()]
    l2 = [safe_log(float(i)) for i in df['l2'].tolist()]
    lr = [safe_log(float(i)) for i in df['lr'].tolist()]
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-5.1, 0.1)
    ax.set_ylim(-5.1, 0.1)
    p = ax.scatter(l1, l2, lr, c=acccuracy, marker='o', cmap='brg', vmin=0, vmax=1)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter_xy))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter_xy))
    ax.zaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter_z))
    ax.set_xticks([safe_log(i) for i in [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]])
    ax.set_yticks([safe_log(i) for i in [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]])
    ax.set_zticks([safe_log(i) for i in [0.00001, 0.0001, 0.001, 0.01]])
    ax.set_xlabel('L1 regularization')
    ax.xaxis.labelpad = 20
    ax.set_ylabel('L2 regularization')
    ax.yaxis.labelpad = 20
    ax.set_zlabel('Learning rate')
    ax.zaxis.labelpad = 20
    plt.title('Hyperparameter search with the {} model \n using {} loss on {}'.format(model, loss, dataset)
              , fontsize=16, fontstyle='italic', fontweight='bold', y=1.08)
    # fig.colorbar(p)
    plt.show()


def log_tick_formatter_z(val, pos=None):
    return 10 ** val


def log_tick_formatter_xy(val, pos=None):
    if val == -5:
        return 0
    return 10 ** val


def get_best_metrics(args, best, results):
    str_list = []
    filter_loss = ['crossentropy', 'hinge', 'mse']
    filter_model = ['cnn', 'mlp', 'linear']
    filter_dataset = ['fashion', 'cifar']
    for dataset in filter_dataset:
        curr_dataset = results['dataset'] == dataset
        curr_dataset = results[curr_dataset]
        for loss in filter_loss:
            curr_loss = curr_dataset['loss_fn'] == loss
            curr_loss = curr_dataset[curr_loss]
            for model in filter_model:
                curr_model = curr_loss['model'] == model
                curr_model = curr_loss[curr_model]
                generate_accuracy_plot(curr_model, dataset, loss, model)

                l1 = curr_model.query('l1!=0 and l2==0')
                res1 = l1.sort_values(by='acc', ascending=False).head(n=1)
                l2 = curr_model.query('l2!=0 and l1==0')
                res2 = l2.sort_values(by='acc', ascending=False).head(n=1)
                l1l2 = curr_model.query('l1!=0 and l2!=0')
                res12 = l1l2.sort_values(by='acc', ascending=False).head(n=1)
                alone = curr_model.query('l1==0 and l2==0')
                resa = alone.sort_values(by='acc', ascending=False).head(n=1)

                best = best.append(res1)
                best = best.append(res2)
                best = best.append(res12)
                best = best.append(resa)
                string1 = 'model__' + res1['model'].values[0] + \
                          '___dataset__' + res1['dataset'].values[0] + \
                          '___loss__' + res1['loss_fn'].values[0] + \
                          '___learning_rate__' + res1['lr'].values[0] + \
                          '___batch_size__' + res1['bs'].values[0] + \
                          '___num_epochs__' + res1['epochs'].values[0] + \
                          '___weight_decay__' + str(res1['l2'].values[0]) + \
                          '___l1_reg__' + str(res1['l1'].values[0])
                string1 = string1.replace('__0.0___', '__0___')
                if string1[-3:] == '0.0':
                    string1 = string1.replace('___l1_reg__0.0', '___l1_reg__0')

                string2 = 'model__' + res2['model'].values[0] + \
                          '___dataset__' + res2['dataset'].values[0] + \
                          '___loss__' + res2['loss_fn'].values[0] + \
                          '___learning_rate__' + res2['lr'].values[0] + \
                          '___batch_size__' + res2['bs'].values[0] + \
                          '___num_epochs__' + res2['epochs'].values[0] + \
                          '___weight_decay__' + str(res2['l2'].values[0]) + \
                          '___l1_reg__' + str(res2['l1'].values[0])
                string2 = string2.replace('__0.0___', '__0___')

                if string2[-3:] == '0.0':
                    string2 = string2.replace('___l1_reg__0.0', '___l1_reg__0')

                string3 = 'model__' + res12['model'].values[0] + \
                          '___dataset__' + res12['dataset'].values[0] + \
                          '___loss__' + res12['loss_fn'].values[0] + \
                          '___learning_rate__' + res12['lr'].values[0] + \
                          '___batch_size__' + res12['bs'].values[0] + \
                          '___num_epochs__' + res12['epochs'].values[0] + \
                          '___weight_decay__' + str(res12['l2'].values[0]) + \
                          '___l1_reg__' + str(res12['l1'].values[0])
                string3 = string3.replace('__0.0___', '__0___')

                if string3[-3:] == '0.0':
                    string3 = string3.replace('___l1_reg__0.0', '___l1_reg__0')

                string4 = 'model__' + resa['model'].values[0] + \
                          '___dataset__' + resa['dataset'].values[0] + \
                          '___loss__' + resa['loss_fn'].values[0] + \
                          '___learning_rate__' + resa['lr'].values[0] + \
                          '___batch_size__' + resa['bs'].values[0] + \
                          '___num_epochs__' + resa['epochs'].values[0] + \
                          '___weight_decay__' + str(resa['l2'].values[0]) + \
                          '___l1_reg__' + str(resa['l1'].values[0])
                string4 = string4.replace('__0.0___', '__0___')
                if string4[-3:] == '0.0':
                    string4 = string4.replace('___l1_reg__0.0', '___l1_reg__0')

                str_list.append(string1)
                str_list.append(string2)
                str_list.append(string3)
                str_list.append(string4)

    for subdir in str_list:
        if os.path.isdir(os.path.join(args.parent_dir, subdir)):
            if not os.path.isdir(os.path.join(args.best_dir, subdir)):
                shutil.copytree(os.path.join(args.parent_dir, subdir),
                                os.path.join(args.best_dir, subdir))
            # shutil.copytree('experiments/grid_search/results/' + subdir, args.experi + subdir)
    return best


if __name__ == "__main__":
    args = parser.parse_args()
    results = pd.DataFrame(columns=['model', 'dataset', 'loss_fn', 'lr', 'bs', 'epochs', 'l1', 'l2', 'acc', 'loss'])
    # Aggregate metrics from args.parent_dir directory
    metrics = dict()
    results = aggregate_metrics(args.parent_dir, metrics)
    results[['l1', 'l2']] = results[['l1', 'l2']].apply(pd.to_numeric)
    best_models = pd.DataFrame(columns=['model', 'dataset', 'loss_fn', 'lr', 'bs', 'epochs', 'l1', 'l2', 'acc', 'loss'])
    best_models = get_best_metrics(best_models, results)

    # writer = pd.ExcelWriter('output.xlsx')
    # best_models.to_excel(writer)
    # writer.save()
