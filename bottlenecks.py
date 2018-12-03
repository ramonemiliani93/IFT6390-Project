import argparse
import logging
import os
import json
import sys
from subprocess import check_call
import pandas as pd
from synthesize_results import get_best_metrics, aggregate_metrics

parser = argparse.ArgumentParser()
parser.add_argument('--parent_dir', default='experiments/grid_search/results',
                    help='Directory containing results of experiments')

PYTHON = sys.executable
if __name__ == '__main__':
    args = parser.parse_args()
    results = pd.DataFrame(columns=['model', 'dataset', 'loss_fn', 'lr', 'bs', 'epochs', 'l1', 'l2', 'acc', 'loss'])

    metrics = dict()
    results = aggregate_metrics(args.parent_dir, metrics)

    results[['l1', 'l2']] = results[['l1', 'l2']].apply(pd.to_numeric)

    best_models = pd.DataFrame(columns=['model', 'dataset', 'loss_fn', 'lr', 'bs', 'epochs', 'l1', 'l2', 'acc', 'loss'])
    best_models = get_best_metrics(best_models, results)
    best_models[['l1', 'l2', 'lr', 'epochs', 'bs']] = best_models[['l1', 'l2', 'lr', 'epochs', 'bs']].apply(
        pd.to_numeric)
    best_models_dir = os.path.join('experiments', 'best_models')
    bottleneck_dir = os.path.join('experiments', 'bottlenecks')

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

        if not os.path.isdir(os.path.join(bottleneck_dir, subdir)):
            exp_dir = os.path.join(bottleneck_dir, subdir)
            os.mkdir(exp_dir)
            with open(os.path.join(exp_dir, 'params.json'), 'w') as outfile:
                json.dump(json_params, outfile)

            cmd = "{python} train.py {} {} {} --bottleneck --model_dir={model_dir} --data_dir {data_dir}".format(
                row['model'],
                row['dataset'],
                row['loss_fn'],
                python=PYTHON,
                model_dir=exp_dir,
                data_dir='data')

            print(cmd)
            check_call(cmd, shell=True)
