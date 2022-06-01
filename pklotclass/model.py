############################################################################################
#                                                                                          #
#                                        Model classes                                     #
#                                                                                          #
#                                   Lionel Cheng, 01.06.2022                               #
#                                                                                          #
############################################################################################
import torch
from torch import nn
from abc import abstractmethod
import numpy as np

class BaseModel(nn.Module):
    """
    Base class for all models. Overrides __str__ method and forces reimplementation of forward.
    """
    @abstractmethod
    def forward(self, *inputs):
        """ Forward pass logic """
        raise NotImplementedError

    @property
    def nparams(self):
        """ Number of trainable parameters of the model"""
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return params

    def __str__(self):
        """ Models prints with number of trainable parameters """
        return super().__str__() + '\nTrainable parameters: {}'.format(self.nparams)

class AlexNet(BaseModel):
    """ Adaptation of the AlexNet convolutional network from Krizhevsky et al. 2012
    for any number of classes assuming input of size 248 x 248. The Dense layers
    have less parameters than the original AlexNet (2048 against 4096). """
    def __init__(self, num_classes=2, init_weights=False):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 48, 11, 4, 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(48, 128, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(128, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(128 * 6 * 6, 2048),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5),
        nn.Linear(2048, 2048),
        nn.ReLU(inplace=True),
        nn.Linear(2048, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class mAlexNet(BaseModel):
    """ Adaptation of the mAlexNet convolutional network from Amato 2017
    for any number of classes assuming input of size 248 x 248"""
    def __init__(self, num_classes=2):
        super().__init__()
        self.input_channel = 3
        self.num_classes = num_classes
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=self.input_channel, out_channels= 16, kernel_size= 11, stride= 4),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(5, 1e-4, 0.75, 2),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels= 16, out_channels= 20, kernel_size= 5, stride= 1),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(5, 1e-4, 0.75, 2),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels= 20, out_channels= 30, kernel_size= 3, stride= 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.layer4 = nn.Sequential(
            nn.Linear(30*3*3, out_features=48),
            nn.ReLU(inplace=True)
        )
        self.layer5 = nn.Sequential(
            nn.Linear(in_features=48, out_features=self.num_classes)
        )

    def forward(self, x):
        x = self.layer3(self.layer2(self.layer1(x)))
        x = x.view(x.size(0), -1)
        x = self.layer5(self.layer4(x))
        return x

class LeNet5(BaseModel):
    """ Adaptation of the LNet5 convolutional network from Lecun et al. 1998
    for any number of classes assuming input of size 248 x 248"""
    def __init__(self, num_classes=2):
        super().__init__()
        self.input_channel = 3
        self.num_classes = num_classes
        self.layer1 = nn.Sequential(
            nn.Conv2d(self.input_channel, 6, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.fc = nn.Linear(16 * 53 * 53, 120)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(120, 84)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(84, self.num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        x = self.relu(x)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        return x

class ResBlock(nn.Module):
    """ Block use for ResNet. """
    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)
        x += identity
        x = self.relu(x)
        return x

class ResNet18(nn.Module):
    """ Adaptation of the ResNet5 convolutional network from He 2015
    for any number of classes assuming input of size 248 x 248"""
    def __init__(self, image_channels, num_classes):

        super(ResNet18, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        #resnet layers
        self.layer1 = self.__make_layer(64, 64, stride=1)
        self.layer2 = self.__make_layer(64, 128, stride=2)
        self.layer3 = self.__make_layer(128, 256, stride=2)
        self.layer4 = self.__make_layer(256, 512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def __make_layer(self, in_channels, out_channels, stride):

        identity_downsample = None
        if stride != 1:
            identity_downsample = self.identity_downsample(in_channels, out_channels)

        return nn.Sequential(
            ResBlock(in_channels, out_channels, identity_downsample=identity_downsample, stride=stride),
            ResBlock(out_channels, out_channels)
        )

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x

    def identity_downsample(self, in_channels, out_channels):

        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels)
        )