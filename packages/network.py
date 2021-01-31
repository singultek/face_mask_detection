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
import shutil
import matplotlib.pyplot as plt
import numpy as np
import os


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
                    transforms.RandomRotation(20),
                    transforms.ToTensor(),
                    # torchvision.transforms.Normalize converts mean and std to torch.tensor
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])]),
                'eval': transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    # torchvision.transforms.Normalize converts mean and std to torch.tensor
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])])
            }

        elif backbone == "BasicCNN" and backbone is not None:
            # Designed BasicCNN is used
            self.net = nn.Sequential(
                # [3, 256, 256] -> Input with 256*256 RGB Image
                nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=1),
                # [64, 252, 252] -> Result of first convolutional layer
                nn.ReLU(inplace=True),
                # [64, 252, 252] -> Result of first ReLU activation function
                nn.MaxPool2d(kernel_size=7, stride=7, padding=0),
                # [64, 36, 36] -> Result of first max pooling operation
                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
                # [128, 36, 36] -> Result of second convolutional layer
                nn.ReLU(inplace=True),
                # [128, 36, 36] -> Result of second ReLU activation function
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                # [128, 18, 18] -> Result of second max pooling operation
                nn.Flatten(),
                # [128*18*18] -> Result of flattening with vector size of 128*18*18
                nn.Linear(128 * 18 * 18, 512),
                # [512] -> Creating a linear layer with flattened vector size
                nn.ReLU(inplace=True),
                # [512] -> Result of ReLU activation function
                nn.Linear(512, 64),
                # [64] -> Creating another linear layer. The reason we have 2 steps for converting
                # flatten layer to linear layer is that we try to decrease dependencies of neurons
                # and get better classifier
                nn.ReLU(inplace=True),
                # [64] -> Result of ReLU activation function
                nn.Dropout(),
                # [64] -> Result of Drop-out operation
                nn.Linear(64, self.number_output)
                # [self.number_output] -> Final output classes
            )

            # Preprocessing the data for BasicCNN input
            # While normalization, mean and standard deviation is zero and one
            self.data_preprocess = {
                'train': transforms.Compose([
                    transforms.RandomResizedCrop(256, scale=(0.8, 1.2), ratio=(3. / 4., 4. / 3.)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(20),
                    transforms.ToTensor(),
                    # torchvision.transforms.Normalize converts mean and std to torch.tensor
                    transforms.Normalize(mean=[0.0, 0.0, 0.0],
                                         std=[1.0, 1.0, 1.0])]),
                'eval': transforms.Compose([
                    transforms.Resize(324),
                    transforms.CenterCrop(256),
                    transforms.ToTensor(),
                    # torchvision.transforms.Normalize converts mean and std to torch.tensor
                    transforms.Normalize(mean=[0.0, 0.0, 0.0],
                                         std=[1.0, 1.0, 1.0])])
            }
        elif backbone is None:
            raise ValueError("The backbone input is not given! Backbone should be selected from {ResNet, BasicCNN}")
        else:
            raise ValueError("The given backbone {} is not recognised".format(str(backbone)))

        # Move network to the selected device's memory
        self.net.to(self.device)

        return

    @staticmethod
    def save_checkpoint(checkpoint: dict,
                        network_path: str,
                        best_network_path: str,
                        is_best_network: bool) -> None:
        """
        Save the checkpoint parameters(network, epochs, optimizer and loss parameters)
        Args:
            checkpoint: the dictionary that stores the saved parameters
            network_path: the current network's file path
            best_network_path: the best network's file path
            is_best_network: the boolean variable to indicate whether current network is best or not
        Returns:
            None
        """
        # Save the given checkpoint parameters onto given network path
        torch.save(checkpoint, network_path)
        # Check the condition of best network
        if is_best_network:
            # Modify the best network file with using shutil module
            shutil.copy(network_path, best_network_path)

        return

    def load_checkpoint(self,
                        saved_network_path: str,
                        optimizer) -> (nn.Sequential, torch.optim, int, float):
        """
        Load the network, epochs, optimizer and loss parameters.
        To load the items, initialization of network and optimizer is the first thing to do.
        After that, rest of the parameters can be loaded from checkpoint dictionary.
        Args:
            saved_network_path: the path where the loading network file is be saved
            optimizer: the optimizer for initialization from the checkpoint
        Returns:
            self.net: the network that is loaded from saved network file
            optimizer: the optimizer parameter that is loaded from saved network file
            epochs: the epoch parameter that is loaded from saved network file
            loss: loss parameter that is loaded from saved network file
        """
        # Load the checkpoint parameters from saved networks path
        checkpoint = torch.load(saved_network_path)
        # Initialize the network and optimizer first
        self.net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # Load the other parameters
        epochs = checkpoint['epoch']
        loss = checkpoint['loss']

        return self.net, optimizer, epochs, loss

    def forward(self,
                x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        The forward stage of Convolutional Neural Network
        Args:
            x: Input tensor of Convolutional Neural Network
        Returns:
            logits: Output before the activation function. logits will be used to compute loss function more precisely.
            outputs: Output after the activation function
        """

        logits = self.net(x)
        outputs = F.softmax(logits, dim=1)

        return logits, outputs

    @staticmethod
    def __decision(outputs: torch.Tensor) -> torch.Tensor:
        """
        The method which applies the argmax operation to give the final decisions of classifier.
        Since argmax is applied, the decision(winnind class ID) will be maximum of the outputs of neurons.
        Args:
            outputs: The outputs of Convolutional Neural Network
        Returns:
            decisions: The decisions(winning class ID) for each example of dataset
        """
        decisions = torch.argmax(outputs, dim=1)

        return decisions

    @staticmethod
    def __loss(logits: torch.Tensor,
               labels: torch.Tensor) -> torch.Tensor:
        """
        The method which applies the Cross Entropy loss funtion to given logits and labels.
        logits will be used to compute loss function more precisely.
        Args:
            logits: Output before the activation function
            labels: Class labels from dataset
        Returns:
            loss: The value of the loss function
        """
        loss = F.cross_entropy(logits, labels, reduction='mean')
        return loss

    @staticmethod
    def __performance(outputs: torch.Tensor,
                      labels: torch.Tensor) -> float:
        """
        The method which computes the accuracy of the networks with given outputs and labels.
        Args:
            outputs: Output after the activation (softmax) function
            labels: Class labels from dataset
        Returns:
            network_accuracy: The value of the prediction accuracy of the network
        """
        # Decisions of CNN Classifier
        decisions = CNNClassifier.__decision(outputs)
        # Checking the equality of decisions and labels
        correct_predictions = torch.eq(decisions, labels)
        # Converting returns of torch.eq(which is boolean) to torch.float and calculating accuracy
        # torch.mean gets a tensor as an input and returns one element tensor, which is mean of correct predictions
        # torch.Tensor.item gets a one element tensor as an input and returns Python float number
        network_accuracy = torch.mean(correct_predictions.to(torch.float) * 100.00).item()
        return network_accuracy

    @staticmethod
    def __plot(network_name: str,
               train_accuracy: np.array,
               validation_accuracy: np.array) -> None:
        """
        The method that visualize the results of training and validation accuracy
        Args:
            network_name: The string that indicates the name of netwrok to save
            train_accuracy: The array that includes the training accuracies
            validation_accuracy: The array that includes the validation accuracies
        Returns:
            None
        """
        # Create the plot for train and validation accuracy results
        plt.plot(train_accuracy, label='Training Data')
        plt.plot(validation_accuracy, label='Validation Data')
        plt.ylabel('Accuracy %')
        plt.xlabel('Epochs')
        plt.grid(True)
        # Splitting the network name for creating title for plot
        plt.title('Backbone {}, # Batch {}, # Epochs {}, Learning Rate {}'.format(
            str(network_name.split('-')[0][:-1]),
            int(network_name.split('-')[1][:-1]),
            int(network_name.split('-')[2][:-1]),
            float(network_name.split('-')[3][:-1])))
        # Declaire the location of legend
        plt.legend(loc='lower  right')

        # Getting currect script's directory and create a directory for saving the results
        script_base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        saved_results_dir = os.path.join(script_base_dir, 'results/')
        # Checking whether there is a forder for results or not
        if not os.path.isdir(saved_results_dir):
            os.makedirs(saved_results_dir)
        # Save the resulted plot as png file
        plt.savefig("{}.png".format(os.path.join(saved_results_dir, network_name)))
        return

    def train_model(self):

        return

    def eval_model(self):
        return

    def classify_input(self):
        return
