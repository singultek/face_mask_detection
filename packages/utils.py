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

# Import the necessary libraries
import argparse
import cv2

from .network import *
from .dataset import *
import torch.utils.data


def face_detection() -> None:
    """
    The method that captures the live image from webcam and classify this input image with pre-trained network
    Args:

    Returns:
        None
    """
    # Create a capture of webcam (by default 0 is webcam, if external camera will be used, 0 should be changed with 1)
    cap = cv2.VideoCapture(0)
    # Load the Haar Cascade filter from opencv module
    face_cascade_filter = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # Start the live video
    while True:
        # Read the frame
        _, image = cap.read()
        # Flip the image for acting like a mirror
        image = cv2.flip(image, 1, 1)
        # Convert original image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Detect the faces with pre-build detectMultiScale method
        faces = face_cascade_filter.detectMultiScale(gray, 1.3, 4)
        # Draw the rectangle on each detected face
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # Display the findings
        cv2.imshow('LIVE FACE DETECTION', image)
        # Live capturing will be stopped when user presses the ESC key
        if cv2.waitKey(1) == 27:
            break

    # Release the live capturing video
    cap.release()
    # Destroy all the windows
    cv2.destroyAllWindows()
    return


def parse_arguments() -> argparse.Namespace:
    """
    The method is created for enhancing user interface on the command line by parsing
    command line arguments.

    Args:

    Returns:
        The Namespace object which holds the input arguments given from command line
    """

    # Creating subparsers for each working mode
    parser = argparse.ArgumentParser(description='The command line argument parser for Face Mask Detector')
    subparsers_set = parser.add_subparsers(title='Working Mode of Network',
                                           description='Main 3 modes of selecting how the network work',
                                           dest='mode',
                                           required=True,
                                           help='Decide the working mode from following options: '
                                                'train = Train the model, '
                                                'evaluate = Evaluate the model, '
                                                'classify = Classify the input with pretrained model')

    # Adding parsers for train mode
    train_parser = subparsers_set.add_parser('train',
                                             description='Train/re-train the network',
                                             help='Train/re-train the network and save it')
    evaluate_parser = subparsers_set.add_parser('evaluate',
                                                description='Evaluate/re-evaluate the network',
                                                help='Evaluate/re-evaluate the network to see the statistical summary of results and save it')
    classify_parser = subparsers_set.add_parser('classify',
                                                description='Classify the input image',
                                                help='Classify the input image which is provided from webcam or folder')

    train_parser.add_argument('dataset_path',
                              type=str,
                              help='A dataset folder to train')
    train_parser.add_argument('--backbone',
                              default='ResNet',
                              type=str,
                              choices=['ResNet', 'BasicCNN'],
                              help='(default = ResNet) A neural network which will be used to train')
    train_parser.add_argument('--resnet_train_mode',
                              default='not_retrain',
                              type=str,
                              choices=['not_retrain', 'retrain'],
                              help='(default = not_retrain) A retraining mode for ResNet backbone')
    train_parser.add_argument('--batch_size',
                              default=64,
                              type=int,
                              help='(default = 64) The size of mini-batch')
    train_parser.add_argument('--epochs',
                              default=10,
                              type=int,
                              help='(default = 10) The number of training epochs')
    train_parser.add_argument('--split_data',
                              nargs=3,
                              default=[0.7, 0.15, 0.15],
                              type=str,
                              help='(default = [0.8,0.2]) The dataset proportions for training process')
    train_parser.add_argument('--learning_rate',
                              default=0.001,
                              type=float,
                              help='(default = 0.001) The learning rate for training with ADAM')
    train_parser.add_argument('--number_workers',
                              default=3,
                              type=int,
                              help='(default = 3) The number of unit which will work during loading data stage')
    train_parser.add_argument('--device',
                              default='cpu',
                              type=str,
                              help='(default = cpu) The device which will be used to compute the neural network {cpu, cuda:0, cuda:1, ...}')
    train_parser.add_argument('--shuffle',
                              default=True,
                              type=bool,
                              help='(default = True) The boolean value which indicates whether data will be randomly shuffled or not')

    # Adding parsers for evaluate mode
    evaluate_parser.add_argument('network_path',
                                 type=str,
                                 help='A network file path to evaluate')
    evaluate_parser.add_argument('dataset_path',
                                 type=str,
                                 help='A dataset folder to evaluate')
    evaluate_parser.add_argument('--backbone',
                                 default='ResNet',
                                 type=str,
                                 choices=['ResNet', 'BasicCNN'],
                                 help='(default = ResNet) A neural network which will be used to evaluate')
    evaluate_parser.add_argument('--resnet_train_mode',
                                 default='not_retrain',
                                 type=str,
                                 choices=['not_retrain', 'retrain'],
                                 help='(default = not_retrain) A retraining mode for ResNet backbone')
    evaluate_parser.add_argument('--batch_size',
                                 default=64,
                                 type=int,
                                 help='(default = 64) The size of mini-batch')
    evaluate_parser.add_argument('--number_workers',
                                 default=3,
                                 type=int,
                                 help='(default = 3) The number of unit which will work during loading data stage')
    evaluate_parser.add_argument('--device',
                                 default='cpu',
                                 type=str,
                                 help='(default = cpu) The device which will be used to compute the neural network {cpu, cuda:0, cuda:1, ...}')
    evaluate_parser.add_argument('--shuffle',
                                 default=False,
                                 type=bool,
                                 help='(default = True) The boolean value which indicates whether data will be randomly shuffled or not')

    # Adding parsers for classify mode
    classify_parser.add_argument('input_path',
                                 default='None',
                                 type=str or None,
                                 help='(default = None) The input path to be used for classification process.'
                                      'If input image path is not specified, webcam will be used by default')
    classify_parser.add_argument('network_path',
                                 type=str,
                                 help='A network file path to classify the input')
    classify_parser.add_argument('--device',
                                 default='cpu',
                                 type=str,
                                 help='(default = cpu) The device which will be used to compute classification process {cpu, cuda:0, cuda:1, ...}')

    args_parsed = parser.parse_args()

    # For train mode, we need to check the split_data input since sum of split_data list should be exactly 1.0
    try:
        split_sum = 0
        for element in args_parsed.split_data:
            try:
                split_sum += float(element)
            except ValueError:
                raise ValueError("Invalid dataset split input. Please try to use proper format, like 0.7 0.15 0.15")

        if split_sum != 1.0:
            raise ValueError(
                "Invalid dataset split input. The sum of proportions of split should be exactly 1.0, like 0.7 0.15 0.15")
    # There is no split_data attribute belongs to Namespace when we don't use training mode. Thus, we will except AttributeError but we can just pass that.
    except AttributeError:
        pass

    return args_parsed


def training(dataset_path: str,
             backbone: str,
             resnet_retrain_mode: str,
             batch_size: int,
             epochs: int,
             split_data: list,
             learning_rate: float,
             number_workers: int,
             device: str,
             shuffle: bool) -> None:
    """
    The main training method to perform the training process of the network with input arguments
    Args:
        dataset_path: the string with the path of dataset
        backbone: the string with the name of network to be used
        resnet_retrain_mode: the string that indicates the choice for retraining on the ResNet training mode
        batch_size: the integer which indicates the element processed at each mini-batch
        epochs: the integer which indicates the number of epochs
        split_data: the list with the training and testing proportions of dataset
        learning_rate: the float that indicates the learning rate of ADAM
        number_workers: the integer that gives the number of working unit while loading data
        device: the string that declares the device used to execute the process
        shuffle: The boolean value which indicates whether data will be randomly shuffled or not

    Returns:
        None
    """
    # Creating a new classifier
    classifier = CNNClassifier(backbone, resnet_retrain_mode, device)
    # Dataset preperation
    dataset = Dataset(dataset_path)
    # Dataset splits
    [train_set, val_set, test_set] = dataset.split_into_test_train(split_data)
    # Defining data preprocess operations
    train_set.preprocess_operation(classifier.data_preprocess['train'])
    val_set.preprocess_operation(classifier.data_preprocess['eval'])
    test_set.preprocess_operation(classifier.data_preprocess['eval'])
    # Converting datasets into data loaders
    train_set = train_set.data_loader(batch_size=batch_size, shuffle=shuffle, number_workers=number_workers)
    val_set = val_set.data_loader(batch_size=batch_size, shuffle=shuffle, number_workers=number_workers)
    test_set = test_set.data_loader(batch_size=batch_size, shuffle=shuffle, number_workers=number_workers)
    # Print some inside information of the data
    dataset.summary_data_characteristics([train_set, val_set, test_set], split_data)
    # Train the classifier
    print('\nTraining stage has started..\n')
    classifier.train_network(train_set, val_set, backbone, resnet_retrain_mode, batch_size, learning_rate, epochs)
    # Load the best resulted model for validation
    print('\nLoading the best model found during trainig..\n')
    network_name = '{}({})-{}-{}-{}'.format(backbone, resnet_retrain_mode, batch_size, epochs, learning_rate)
    filepath = '{}.pth'.format(os.path.join('./models/', network_name))
    classifier.load(saved_network_path=filepath)
    print('\nValidation stage has started..\n')
    train_acc = classifier.eval_network(train_set)
    val_acc = classifier.eval_network(val_set)
    test_acc = classifier.eval_network(test_set)
    print('\nAccuracies of {} in the {}\n'.format(network_name, filepath))
    print('Training set:\t{}'.format(round(train_acc, 4)))
    print('Validation set:\t{}'.format(round(val_acc, 4)))
    print('Test set:\t{}'.format(round(test_acc, 4)))
    return


def evaluating(network_path: str,
               dataset_path: str,
               backbone: str,
               resnet_retrain_mode: str,
               batch_size: int,
               number_workers: int,
               device: str,
               shuffle: bool) -> None:
    """
    The main evaluating method to perform the evaluation process of the network with input arguments
    Args:
        network_path: the string that provides the file path of the saved network
        dataset_path: the string with the path of dataset
        backbone: the string with the name of network to be used
        resnet_retrain_mode: the string that indicates the choice for retraining on the ResNet training mode
        batch_size: the integer which indicates the element processed at each mini-batch
        number_workers: the integer that gives the number of working unit while loading data
        device: the string that declares the device used to execute the process
        shuffle: The boolean value which indicates whether data will be randomly shuffled or not
    Returns:
        None
    """
    # Creating a new classifier
    classifier = CNNClassifier(backbone, resnet_retrain_mode, device)
    # Dataset preperation
    dataset = Dataset(dataset_path)
    # Defining data preprocess operations
    dataset.preprocess_operation(classifier.data_preprocess['eval'])
    # Converting datasets into data loaders
    dataset = dataset.data_loader(batch_size=batch_size, shuffle=shuffle, number_workers=number_workers)

    print('\nLoading the model to evaluate..\n')
    classifier.load(saved_network_path=network_path)
    print('\nValidation stage has started..\n')
    acc = classifier.eval_network(dataset)
    print('\nAccuracy: {} of the network in the {}\n'.format(round(acc, 4), network_path))
    return


def classifying(input_path: str,
                network_path: str,
                device: str) -> None:
    """
    The main classifying method to perform the classification process of the network with input arguments
    Args:
        input_path: the string that states file path of the input image which will be classified
        network_path: the string that provides the file path of the saved network
        device: the string that declares the device used to execute the process
    Returns:
        None
    """

    print(input_path, network_path, device)
    return
