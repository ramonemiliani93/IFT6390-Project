from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, FashionMNIST

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
        data_set = FashionMNIST(root=root, train=train, download=download, transform=fashion_transformer)
    else:
        pass
    return data_set


def fetch_train_dataloaders(dataset, data_dir, params, split=0.2):
    """
    Fetches dataloaders for train and test, choosing from the indicated datset.
    :param dataset: datset to be loaded (cifar|fashion).
    :param data_dir: path where images will be saved or, ifa already downloaded, current path of dataset
    :param params: hyperparameters
    :param split: portion of data used for validation (0-1)
    :return: train and validation dataloaders
    """

    # Load datasets, in case data does not exist then download it
    try:
        set = create_dataset(dataset, data_dir, True, False)
    except RuntimeError:
        set = create_dataset(dataset, data_dir, True, True)

    # Split
    val_size = int(split * len(set))
    train_size = len(set) - val_size
    train_set, val_set = random_split(set, [train_size, val_size])

    train_dl = DataLoader(train_set, batch_size=params.batch_size, shuffle=True, num_workers=NUM_WORKERS,
                          pin_memory=params.cuda)
    val_dl = DataLoader(val_set, batch_size=params.batch_size, shuffle=True, num_workers=NUM_WORKERS,
                        pin_memory=params.cuda)

    return train_dl, val_dl


def fetch_test_dataloader(dataset, data_dir, params):
    """
    Fetches dataloaders for test, choosing from the indicated dataset.
    :param dataset: datset to be loaded (cifar|fashion).
    :param data_dir: path where images will be saved or, ifa already downloaded, current path of dataset
    :param params: hyperparameters
    :return: test dataloader
    """

    # Load datasets, in case data does not exist then download it
    try:
        test_set = create_dataset(dataset, data_dir, False, False)
    except RuntimeError:
        test_set = create_dataset(dataset, data_dir, False, True)

    test_dl = DataLoader(test_set, batch_size=params.batch_size, shuffle=True, num_workers=NUM_WORKERS,
                         pin_memory=params.cuda)

    return test_dl
