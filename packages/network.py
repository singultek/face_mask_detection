""" Copyright 2021 - Sinan Gültekin <singultek@gmail.com>

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# Importing necessary libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torchvision import transforms


class CNNClassifier(nn.Module):
    """
    The Class which is responsible from the classification task with given input.
    Inherited from torch.nn.Module
    """

    def __init__(self,
                 backbone='ResNet',
                 resnet_retrain_mode='not_retrain',
                 device='cpu') -> None:
        """
        Initializing the classifier
        Args:
            backbone: the string with the name of network to be used.
                    By default it is ResNet and user can select an option from {ResNet, BasicCNN}
            resnet_retrain_mode: the string that indicates the choice for retraining on the ResNet training mode
                    By default it is not_retrain and user can select an option from {not_retrain, retrain}
            device: the string that declares the device used to execute the process.
                    By default it is cpu and user can select an option from {cpu, cuda:0, cuda:1, ...}
        Return:
            None
        """
        super(CNNClassifier, self).__init__()

        # Class attributes are declared
        self.number_output = 3  # In Face Mask Detection Problem, there are 3 classes
        self.net = None  # The network will be assigned in the next methods
        self.device = torch.device(
            device)  # The attribute to assign the selected device to the network and torch module
        self.data_preprocess = None  # The attribute for applying data augmentation/preprocessing

        if backbone == "ResNet" and backbone is not None:
            # Load the pre-trained ResNet50
            self.net = torch.hub.load('pytorch/vision:v0.6.0', 'resnet50', pretrained=True)
            # Freeze the parameters(network weights and biases) of ResNet
            if resnet_retrain_mode == 'not_retrain' and resnet_retrain_mode is not None:
                for parameter in self.net.parameters():
                    parameter.requires_grad = False
            # Unfreeze the parameters(network weights and biases) of the last 4 layers (layer3, layer4, avgpool, fc) of ResNet
            # In total there are 10 layers, user can change the layer_threshold if they would like to train more or less layers
            elif resnet_retrain_mode == 'retrain' and resnet_retrain_mode is not None:
                counter = 0
                layer_threshold = 6
                for child in self.net.children():
                    counter += 1
                    if counter > layer_threshold:
                        for parameter in child.parameters():
                            parameter.requires_grad = True
            elif resnet_retrain_mode is None:
                raise ValueError(
                    "The ResNet retrain mode input is not given! ResNet retrain mode should be selected from {not_retrain, retrain}")
            else:
                raise ValueError("The given ResNet retrain mode  {} is not recognised".format(str(resnet_retrain_mode)))

            # Modifying the last layer with respect to our output classes
            self.net.fc = nn.Linear(2048, self.number_output)

            # Preprocessing the data for ResNet input
            # While normalization, mean and standard deviation is the original ones which used to train ResNet
            self.data_preprocess = {
                'train': transforms.Compose([
                    transforms.RandomResizedCrop(224, scale=(0.8, 1.2), ratio=(3. / 4., 4. / 3.)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=torch.tensor([0.485, 0.456, 0.406]),
                                         std=torch.tensor([0.229, 0.224, 0.225]))]),
                'eval': transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=torch.tensor([0.485, 0.456, 0.406]),
                                         std=torch.tensor([0.229, 0.224, 0.225]))])
            }

        elif backbone == "BasicCNN" and backbone is not None:
            # Designed BasicCNN is used
            self.net = nn.Sequential(

            )

        elif backbone is None:
            raise ValueError("The backbone input is not given! Backbone should be selected from {ResNet, BasicCNN}")
        else:
            raise ValueError("The given backbone {} is not recognised".format(str(backbone)))
        return

    def save(self):
        return

    def load(self):
        return

    def forward(self):
        return

    @staticmethod
    def __decision(self):
        return

    def train_model(self):
        return

    def eval_model(self):
        return

    def classify_input(self):
        return

    @staticmethod
    def __loss(self):
        return

    @staticmethod
    def __performance(self):
        return

    def __plot(self):
        return
