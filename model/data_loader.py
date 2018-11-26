from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, FashionMNIST, MNIST

# Constants
NUM_WORKERS = 8

# Transform for CIFAR10
cifar_transformer = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize(28),
    transforms.ToTensor()])  # Tensor

# Transform for FashionMNIST
fashion_transformer = transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))])  # Tensor


def create_dataset(dataset, root, train, download):
    data_set = None
    if dataset == 'cifar':
        data_set = CIFAR10(root=root, train=train, download=download, transform=cifar_transformer)
    elif dataset == 'fashion':
        data_set = MNIST(root=root, train=train, download=download, transform=fashion_transformer)
    else:
        pass
    return data_set


def fetch_dataloader(dataset, data_dir, params):
    """
    Fetches dataloaders for train and test, choosing from the indicated datset.
    :param dataset: datset to be loaded (cifar|fashion).
    :param data_dir: path where images will be saved or, ifa already downloaded, current path of dataset
    :param params: hyperparameters
    :return: train and test dataloaders
    """

    # Load datasets, in case data does not exist then download it
    try:
        train_set = create_dataset(dataset, data_dir, True, False)
        val_set = create_dataset(dataset, data_dir, False, False)
    except RuntimeError:
        train_set = create_dataset(dataset, data_dir, True, True)
        val_set = create_dataset(dataset, data_dir, False, False)

    train_dl = DataLoader(train_set, batch_size=params.batch_size, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=params.cuda)
    val_dl = DataLoader(val_set, batch_size=params.batch_size, shuffle=True,
                        num_workers=NUM_WORKERS, pin_memory=params.cuda)

    return train_dl, val_dl
