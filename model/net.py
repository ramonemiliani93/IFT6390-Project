"""Defines the models"""

import torch.nn as nn

# Constants
NUM_CLASSES = 10
IN_FEATURES = 784
LAYERS = [5]


class LinearRegression(nn.Module):
    """
    Softmax regression model
    """

    def __init__(self):
        super(LinearRegression, self).__init__()
        self.num_classes = NUM_CLASSES
        self.in_features = IN_FEATURES
        self.features = nn.Sequential(
            nn.Linear(self.in_features, self.num_classes)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.features(x)
        return x


class MLP(nn.Module):
    """
    Multilayer perceptron model
    """

    def __init__(self):
        super(MLP, self).__init__()
        self.num_classes = NUM_CLASSES
        self.in_features = IN_FEATURES
        self.layers = LAYERS
        # Add input layer and output layer
        self.layers.insert(0, self.in_features)
        self.layers.insert(len(self.layers), self.num_classes)
        self.features = nn.Sequential(
            *[nn.Linear(self.layers[i], self.layers[i+1]) for i in range(len(self.layers)-1)]
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.features(x)
        return x


class CNN(nn.Module):
    """
    CNN model
    """

    def __init__(self):
        super(CNN, self).__init__()
        self.num_classes = NUM_CLASSES
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.classifier = nn.Sequential(
            nn.Linear(256, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, 10))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
