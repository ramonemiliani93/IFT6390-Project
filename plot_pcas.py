import os
import sys
from subprocess import check_call

python = sys.executable
best_models_dir = os.path.join('experiments', 'bottlenecks')

for subdir in os.listdir(best_models_dir):
    if subdir == '.DS_Store':
        continue

    settings = subdir.split('___')
    subsettings = [subset.split('__') for subset in settings]
    pd_settings = [item[1] for item in subsettings]

    cmd = '{} embeddings.py {} {} {}  --model_dir {} --fig_name {} --save_fig pca_figures'.format(python,
                                                                                                  pd_settings[0],
                                                                                                  pd_settings[1],
                                                                                                  os.path.join(
                                                                                                      'experiments',
                                                                                                      'grid_search',
                                                                                                      'results',
                                                                                                      subdir,
                                                                                                      'best.pth.tar'),
                                                                                                  os.path.join(
                                                                                                      'experiments',
                                                                                                      'grid_search',
                                                                                                      'results',
                                                                                                      subdir),
                                                                                                  pd_settings[0] + '_' +
                                                                                                  pd_settings[1] + '_' +
                                                                                                  pd_settings[2])

    print(cmd)

    check_call(cmd, shell=True)
