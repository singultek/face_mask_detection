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
import argparse

def webcam():
    return

def face_detection():
    return


def parse_arguments() -> argparse.Namespace:
    '''
    The method is created for enhancing user interface on the command line by parsing
    command line arguments.

    Args:
        None
    Returns:
        The Namespace object which holds the input arguments given from command line
    '''

    #Creating subparser for each working mode
    parser = argparse.ArgumentParser(description='The command line argument parser for Face Mask Detector')
    subparsers_set = parser.add_subparsers(title='Working Mode of Network',
                                      description='Main 3 modes of selecting how the network work',
                                      dest='mode',
                                      required=True,
                                      help='Decide the working mode from following options: '
                                           'train = Train the model, '
                                           'evaluate = Evaluate the model, '
                                           'classify = Classify the input with pretrained model')

    #Adding parsers for train mode
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
    train_parser.add_argument('--batch_size',
                              default=64,
                              type=int,
                              help='(default = 64) The size of mini-batch')
    train_parser.add_argument('--epochs',
                              default=10,
                              type=int,
                              help='(default = 10) The number of training epochs')
    train_parser.add_argument('--split_data',
                              nargs=2,
                              default=[0.8, 0.2],
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

    #Adding parsers for evaluate mode
    evaluate_parser.add_argument('network_path',
                              type=str,
                              help='A network file path to evaluate')
    evaluate_parser.add_argument('--device',
                              default='cpu',
                              type=str,
                              help='(default = cpu) The device which will be used to compute the neural network {cpu, cuda:0, cuda:1, ...}')

    #Adding parsers for classify mode
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

    #For train mode, we need to check the split_data input since sum of split_data list should be exactly 1.0
    try:
        split_sum = 0
        for element in args_parsed.split_data:
            try:
                split_sum += float(element)
            except ValueError:
                raise ValueError("Invalid dataset split input. Please try to use proper format, like 0.8 0.2")

        if split_sum != 1.0:
            raise ValueError("Invalid dataset split input. The sum of proportions of split should be exactly 1.0, like 0.8 0.2")
    #There is no split_data attribute belongs to Namespace when we don't use training mode. Thus, we will except AttributeError but we can just pass that.
    except AttributeError:
        pass

    return args_parsed

def training():
    return

def evaluating():
    return

def classifying():
    return
