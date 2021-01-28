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


class CNNClassifier(nn.Module):
    """
    The Class which is responsible from the classification task with given input.
    Inherited from torch.nn.Module
    """

    def __init__(self,
                 backbone='ResNet',
                 device='cpu') -> None:
        """
        Initializing the classifier
        Args:
            backbone: the string with the name of network to be used.
                    By default it is ResNet and user can select an option from {ResNet, BasicCNN}
            device: the string that declares the device used to execute the process.
                    By default it is cpu and user can select an option from {cpu, cuda:0, cuda:1, ...}
        Return:
            None
        """
        super(CNNClassifier, self).__init__()

        # Class attributes are declared
        self.number_output = 3  # In Face Mask Detection Problem, there are 3 classes
        self.net = None  # The network will be assigned in the next methods
        self.device = torch.device(device)  # The attribute to assign the selected device to the network and torch module

        if backbone == "ResNet" and backbone is not None:
            pass
        elif backbone == "BasicCNN" and backbone is not None:
            pass
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
