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
    results = pd.DataFrame(columns=['lr', 'l1', 'l2', 'acc', 'loss'])

    for subdir in os.listdir(parent_dir):
        metrics_file = os.path.join(parent_dir, subdir, 'metrics_val_best_weights.json')
        if os.path.isfile(metrics_file):
            settings = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", subdir)
            if settings[1] == '-05':
                pd_settings = [1e-5, settings[-1], settings[-3]]
            else:
                pd_settings = [settings[0], settings[-1], settings[-3]]

            with open(metrics_file, 'r') as f:
                metrics[subdir] = json.load(f)
                pd_settings.append(metrics[subdir]['accuracy'])
                pd_settings.append(metrics[subdir]['loss'])
                pd_settings = np.asarray(pd_settings)
                row = pd.DataFrame(np.expand_dims(pd_settings, 0), columns=['lr', 'l1', 'l2', 'acc', 'loss'])
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
    results = pd.DataFrame(columns=['lr', 'l1', 'l2', 'acc', 'loss'])
    # Aggregate metrics from args.parent_dir directory
    metrics = dict()
    results = aggregate_metrics(args.parent_dir, metrics)
    table = metrics_to_table(metrics)

    # Display the table to terminal
    print(table)

    # Save results in parent_dir/results.md
    save_file = os.path.join(args.parent_dir, "results.md")
    with open(save_file, 'w') as f:
        f.write(table)