"""Evaluates the model"""
import argparse
import logging
import os

import numpy as np
import torch
from torch.autograd import Variable

import utils
from metrics import dict_metrics
from model.net import LinearRegression, MLP, CNN
import model.data_loader as data_loader
from losses import CrossEntropyWithL1Loss, MultiMarginWithL1Loss, MSEWithL1Loss
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data', help="Directory that will contain dataset")
parser.add_argument('--model_dir', default='experiments/grid_search/results', help="Directory containing params.json")
parser.add_argument('--restore_file', default='best', help="name of the file in --model_dir \
                     containing weights to load")


def evaluate(model, loss_fn, dataloader, metrics, params):
    """Evaluate the model on `num_steps` batches.
    Args:
        model: (torch.nn.Module) the neural network
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """
    # Last layer activation
    activation = torch.nn.Softmax(dim=1)

    # set model to evaluation mode
    model.eval()

    # summary for current eval loop
    summ = []

    # compute metrics over the dataset
    for data_batch, labels_batch in dataloader:

        # move to GPU if available
        if params.cuda:
            data_batch, labels_batch = data_batch.cuda(), labels_batch.cuda()
        # fetch the next evaluation batch
        data_batch, labels_batch = Variable(data_batch), Variable(labels_batch)

        # compute model output
        output_batch = model(data_batch)
        loss = loss_fn(output_batch, labels_batch, model)

        # extract data from torch Variable, move to cpu, convert to numpy arrays
        output_batch = activation(output_batch).data.cpu().numpy()
        labels_batch = labels_batch.data.cpu().numpy()

        # compute all metrics on this batch
        summary_batch = {metric: metrics[metric](output_batch, labels_batch)
                         for metric in metrics}
        summary_batch['loss'] = loss.data.item()
        summ.append(summary_batch)

    # compute mean of all metrics in summary
    metrics_mean = {metric: np.mean([x[metric] for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    # logging.info("- Eval metrics : " + metrics_string)
    return metrics_mean, loss


if __name__ == '__main__':
    """
        Evaluate all results on the test set.
    """
    # Load the parameters
    args = parser.parse_args()
    current_path = os.path.join(os.getcwd(), args.model_dir)
    for dir in tqdm(os.listdir(current_path)):
        dir_path = os.path.join(current_path, dir)
        if os.path.isdir(dir_path):
            if dir.split('__')[5] =='triplet':
                continue
            json_path = os.path.join(dir_path, 'params.json')
            assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
            params = utils.Hyperparameters(json_path)

            # use GPU if available
            params.cuda = torch.cuda.is_available()  # use GPU is available

            # Set the random seed for reproducible experiments
            torch.manual_seed(230)
            if params.cuda: torch.cuda.manual_seed(230)

            # Get the logger
            utils.set_logger(os.path.join(dir_path, 'evaluate.log'))

            # Create the input data pipeline
            # logging.info("Creating the dataset...")

            # fetch dataloaders
            test_dl = data_loader.fetch_test_dataloader(dir.split('__')[3], args.data_dir, params)

            # logging.info("- done.")

            # Define the model and optimizer
            models = {
                'linear': LinearRegression().cuda() if params.cuda else LinearRegression(),
                'mlp': MLP().cuda() if params.cuda else MLP(),
                'cnn': CNN().cuda() if params.cuda else CNN()
            }
            model = models[dir.split('__')[1]]

            # fetch loss function and metrics
            losses = {
                'crossentropy': CrossEntropyWithL1Loss(params.l1_reg),
                'hinge': MultiMarginWithL1Loss(params.l1_reg),
                'mse': MSEWithL1Loss(params.l1_reg)
            }
            loss_fn = losses[dir.split('__')[5]]
            metrics = dict_metrics

            # logging.info("Starting evaluation")

            # Reload weights from the saved file
            utils.load_all_checkpoint(os.path.join(dir_path, args.restore_file + '.pth.tar'), model)

            # Evaluate
            test_metrics, loss = evaluate(model, loss_fn, test_dl, metrics, params)
            save_path = os.path.join(dir_path, "metrics_test_{}.json".format(args.restore_file))
            utils.save_dict_to_json(test_metrics, save_path)
