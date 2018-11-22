from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, FashionMNIST

# Constants
NUM_WORKERS = 8

# Transform for training
train_transformer = transforms.Compose([
    transforms.ToTensor()])  # Tensor

# Transform for evaluation, same as training one
val_transformer = transform = transforms.Compose([
    transforms.ToTensor()])  # Tensor


def create_dataset(dataset, root, train, download, transform):
    data_set = None
    if dataset == 'cifar':
        data_set = CIFAR10(root=root, train=train, download=download, transform=transform)
    elif dataset == 'fashion':
        data_set = FashionMNIST(root=root, train=train, download=download, transform=transform)
    else:
        print("ERROR NI EL PERRO")
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

    try:
        train_set = create_dataset(dataset, data_dir, True, False, train_transformer)
        val_set = create_dataset(dataset, data_dir, False, False, val_transformer)
    except RuntimeError:
        train_set = create_dataset(dataset, data_dir, True, True, train_transformer)
        val_set = create_dataset(dataset, data_dir, False, False, val_transformer)

    train_dl = DataLoader(train_set, batch_size=params.batch_size, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=params.cuda)
    val_dl = DataLoader(val_set, batch_size=params.batch_size, shuffle=True,
                        num_workers=NUM_WORKERS, pin_memory=params.cuda)

    return train_dl, val_dl
