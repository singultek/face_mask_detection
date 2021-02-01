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
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image


class Dataset(torch.utils.data.Dataset):
    """
    The Class which is responsible from the accessing the image data, packing data into mini-batches,
    applying transform operation(data preprocessing) and provide data into torch.Tensor format.
    Inherited from torch.utils.data.Dataset
    """

    def __init__(self,
                 data_path: str,
                 empty_dataset: bool = False) -> None:
        """
        Initializing the Dataset
        Args:
            data_path: The string that declares the file path of the dataset
            empty_dataset: The boolean that indicates the choice about filling Dataset class with dataset images
        Returns:
             None
        """
        # Initialize the class attributes
        # Dataset folder path
        self.data_path = data_path
        # List of names of dataset images
        self.files = []
        # Labels of each dataset images
        self.labels = []
        # List of image data
        self.image_data = []
        # Data preprocessing operation to apply on eaxh images
        self.preprocess = None

        # Checking the data_path attribute
        if data_path is None:
            raise ValueError('Path of dataset folder is not given')
        if not os.path.exists(data_path) or os.path.isfile(data_path):
            raise ValueError("Please check the given datapath {} ".format(str(data_path)))

        # Getting the number of output classes
        folders = os.listdir(self.data_path)
        output_classes = [folder for folder in folders if os.path.isdir(os.path.join(self.data_path, folder))
                          and not folder.endswith('.')]
        output_classes = sorted(output_classes)
        self.number_class = len(output_classes)

        # If we have empty_dataset= False, we will load the file names and labes into
        # files and labels empty lists
        if not empty_dataset:
            counter = 0
            # Looping over each output classes
            for each_output_classes in output_classes:
                # For each output classes, we create file path
                output_classes_folder = os.path.join(self.data_path, each_output_classes)
                # For each created file path, we get the list of image files
                output_classes_folder_files = os.listdir(output_classes_folder)
                # Creating files list which consists of each image files path
                files = [os.path.join(output_classes_folder, file) for file in output_classes_folder_files
                         if os.path.isfile(os.path.join(output_classes_folder, file)) and file.endswith('.jpg')]
                # Extending created files path list to class attribute
                self.files.extend(files)
                # Extending file labels to class attribute
                # File labels are created from lenght of files list times that element of corresponding class label
                # As a result, the list consists of number of files times class labels (mask - 0, no mask - 1, wrong mask - 2)
                self.labels.extend([counter] * len(files))
                counter += 1
        return

    def __len__(self) -> int:
        """
        The method which gives the lenght of dataset
        Args:
            None
        Returns:
            lenght of dataset

        """
        return len(self.files)

    def __getitem__(self,
                    index: int) -> tuple:
        """
        The method which gets the indexof item from dataset and givens the tuple of image and label of that image
        Args:
            index: index of item from dataset
        Returns:
            image, label: The tuple pair of image and label of image
        """
        # Load the image using PIL
        image = Image.open(self.files[index]).convert('RGB')

        # Apply the preprocessing operation if it is needed
        if self.preprocess is not None:
            image = self.preprocess(image)  # preprocessing image

        # Get the label in type of torch.tensor
        label = torch.tensor(self.labels[index], dtype=torch.long)
        return image, label

    def preprocess_operation(self,
                             preprocess_operation: transforms or transforms.Compose or nn.Sequential) -> None:
        """
        Args:
            preprocess_operation: The preprocess operation to apply each input image
        Returns:
            None
        """
        self.preprocess = preprocess_operation
        return

    def split_into_test_train(self):
        return

    def data_loader(self):
        return

    def summary_data_characteristics(self):
        return
