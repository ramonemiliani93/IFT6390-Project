"""Aggregates results from the metrics_eval_best_weights.json in a parent folder"""

import argparse
import json
import os
import pandas as pd
from tabulate import tabulate
import re
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--parent_dir', default='experiments/grid_search/results',
                    help='Directory containing results of experiments')


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


if __name__ == "__main__":
    args = parser.parse_args()
    results = pd.DataFrame(columns=['model', 'dataset', 'loss_fn', 'lr', 'bs', 'epochs', 'l1', 'l2', 'acc', 'loss'])
    # Aggregate metrics from args.parent_dir directory
    metrics = dict()
    results = aggregate_metrics(args.parent_dir, metrics)
    # table = metrics_to_table(metrics)
    results[['l1', 'l2']] = results[['l1', 'l2']].apply(pd.to_numeric)
    filter_loss = ['crossentropy', 'hinge', 'mse']
    filter_model = ['cnn', 'mlp', 'linear']
    filter_dataset = ['fashion', 'cifar']

    best = pd.DataFrame(columns=['model', 'dataset', 'loss_fn', 'lr', 'bs', 'epochs', 'l1', 'l2', 'acc', 'loss'])

    for dataset in filter_dataset:
        curr_dataset = results['dataset'] == dataset
        curr_dataset = results[curr_dataset]
        for loss in filter_loss:
            curr_loss = curr_dataset['loss_fn'] == loss
            curr_loss = curr_dataset[curr_loss]
            for model in filter_model:
                curr_model = curr_loss['model'] == model
                curr_model = curr_loss[curr_model]

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

    writer = pd.ExcelWriter('output.xlsx')
    best.to_excel(writer)
    writer.save()
