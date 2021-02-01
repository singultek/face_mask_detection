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

from packages.utils import parse_arguments
from packages.utils import training
from packages.utils import evaluating
from packages.utils import classifying


def main():
    """
    The main method for executing the working mode given by parsing command line arguments

    Args:

    Returns:
        None
    Raises:
        RuntimeError If the given working mode is not a member of the following set {train, evaluate, classify}
    """

    command_line_args = parse_arguments()

    if command_line_args.mode == 'train':
        print("Training the network is executing!")
        training(command_line_args.dataset_path,
                 command_line_args.backbone,
                 command_line_args.resnet_train_mode,
                 command_line_args.batch_size,
                 command_line_args.epochs,
                 command_line_args.split_data,
                 command_line_args.learning_rate,
                 command_line_args.number_workers,
                 command_line_args.device,
                 command_line_args.shuffle)
    elif command_line_args.mode == 'evaluate':
        print("Evaluating the network is executing!")
        evaluating(command_line_args.network_path,
                   command_line_args.dataset_path,
                   command_line_args.backbone,
                   command_line_args.resnet_retrain_mode,
                   command_line_args.batch_size,
                   command_line_args.number_workers,
                   command_line_args.device,
                   command_line_args.shuffle)
    elif command_line_args.mode == 'classify':
        print("Classifying the input is executing!")
        classifying(command_line_args.input_path,
                    command_line_args.network_path,
                    command_line_args.device)
    else:
        raise RuntimeError(
            "To achieve successful execution, one of the following working mode should be selected; {train, evaluate, classify} ")


if __name__ == '__main__':
    main()
