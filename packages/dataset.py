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
import torch.utils.data
from torchvision import transforms
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
        self.output_classes = [folder for folder in folders if os.path.isdir(os.path.join(self.data_path, folder))
                               and not folder.endswith('.')]
        self.output_classes = sorted(self.output_classes)
        self.number_class = len(self.output_classes)

        # If we have empty_dataset = False, we will load the file names and labes into files and labels empty lists
        if not empty_dataset:
            counter = 0
            # Looping over each output classes
            for each_output_classes in self.output_classes:
                # For each output classes, we create file path
                output_classes_folder = os.path.join(self.data_path, each_output_classes)
                # For each created file path, we get the list of image files
                output_classes_folder_files = os.listdir(output_classes_folder)
                # Creating files list which consists of each image files path
                files = [os.path.join(output_classes_folder, file) for file in output_classes_folder_files if os.path.isfile(os.path.join(output_classes_folder, file)) and file.endswith('.jpg')]
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
        The method which gets the index of item from dataset and givens the tuple of image and label of that image
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
        The method that performs data preprocessing into each dataset samples
        Args:
            preprocess_operation: The preprocess operation to apply each input image
        Returns:
            None
        """
        self.preprocess = preprocess_operation
        return

    def split_into_test_train(self,
                              proportions: list) -> list:
        """
        The method which splits the dataset into given proportions
        Args:
            proportions: The proportion of dataset splits Example = [0.7, 0.15, 0.15]
        Returns:
            splitted_datasets
        """
        # Firstly, check the given proportion input is valid or not
        if len(proportions) == 0 and not sum(proportions) == 1:
            raise ValueError('Invalid proportion is given, their lenght must be non 0 and sum must be 1')
        else:
            for prop in proportions:
                if prop <= 0.0:
                    raise ValueError('Invalid proportions is given, each portion must be greater than 0')
        # Then check the files and labels
        if len(self.files) == 0 or len(self.labels) == 0:
            raise RuntimeError('Split operation cannot perfrom on empty dataset lists')

        # Getting number of splits
        number_splits = len(proportions)
        # Creating empty list and dictionary for splitted dataset
        splitted_datasets = []
        self.splitted_datasets_each_class = {}
        # Adding empty list into each key value in the dictionary
        # Example {0:[], 1:[], 2:[], ..}
        for classes in range(0, self.number_class):
            self.splitted_datasets_each_class[classes] = []
        # Adding image file from self.files to dictionary with respect to its label
        # Example: dictionary key -> label and dictionary value -> image file
        for image_file in range(0, len(self.files)):
            self.splitted_datasets_each_class[self.labels[image_file]].append(image_file)
        # Creating Dataset object for each split
        for i in range(0, number_splits):
            splitted_datasets.append(Dataset(self.data_path, empty_dataset=True))

        # Splitting dataset
        for class_j in range(0, self.number_class):
            # Index of considered element starts from 0
            start = 0
            # For each class in the number_class, we split the that classes' dataset into number_splits
            # In this case, each class of dataset (mask, no_mask, wrong_mask(if 3 dataset folder is used)) is devided into 3 proportions. For each proportion(current_split) in the splitted data,
            # the data from each classes are added with respect to proportion(current split) * total dataset of corresponding class.
            # At the end, splitted_datasets has splitted into 3 proportions (train,validation,test) all include all classes of dataset (mask, no_mask, wrong_mask(if 3 dataset folder is used))
            for current_split in range(0, number_splits):
                # Calculating number of element for each class 'class_j'
                n = int(proportions[current_split] * len(self.splitted_datasets_each_class[class_j]))
                # Index of considered element end with following conditions
                end = start + n if current_split < number_splits - 1 else len(self.splitted_datasets_each_class[class_j])
                # Looping over indices to considered at each iteration
                for ids in self.splitted_datasets_each_class[class_j][start:end]:
                    # Appending the selected data and its label to the current split on Dataset object
                    splitted_datasets[current_split].files.append(self.files[ids])
                    splitted_datasets[current_split].labels.append(self.labels[ids])
                # Updating starting index for next iteration
                start = end
        return splitted_datasets

    def data_loader(self,
                    batch_size: int = 64,
                    shuffle: bool = True,
                    number_workers: int = 3) -> torch.utils.data.DataLoader:
        """
        The method that convert datasets into data loaders with the help of torch.utils.data module
        Args:
            batch_size: (default=64) The integer which indicates the element processed at each mini-batch
            shuffle: (default=True) The boolean value that states whether dataset will be shuffled or not
            number_workers: (default=3) The integer that gives the number of working unit while loading data
        Returns:
            dataloader: The converted data loader which will provide data iteratively
        """
        if len(self.files) == 0 or len(self.labels) == 0:
            raise RuntimeError('Converting to data loaders operation cannot perfrom on empty dataset lists')
        dataloader = torch.utils.data.DataLoader(self,
                                                 batch_size=batch_size,
                                                 shuffle=shuffle,
                                                 num_workers=number_workers)
        return dataloader

    def summary_data_characteristics(self,
                                     input_dataset: list,
                                     proportions: list) -> None:
        """
        The method for printing basic characteristics of dataset
        Args:
            input_dataset: The list that consists of Dataset objects
            proportions: The proportions that is used to split dataset
        Returns:
            None
        """
        print('The number of samples:\t{}'.format(len(self.files)))
        print('The labels of classes are {}'.format(self.output_classes))
        print('After splitting dataset with respect to train set: {}, '
              'validation set: {}, test set: {}'.format(proportions[0], proportions[1], proportions[2]))
        print('The dataset proportions has type for train set:{},'
              'validation set: {}, test set: {}'.format(type(input_dataset[0]), type(input_dataset[1]), type(input_dataset[2])))

        for i in range(self.number_class):
            print('The class:   {}( encoded as {} )   ->  {} samples'.format(self.output_classes[i], i, len(self.splitted_datasets_each_class[i])))
            print('The data distributions with respect to labels of classes {}'.format(round(float(len(self.splitted_datasets_each_class[i])/len(self.files)), 3)))
        return
