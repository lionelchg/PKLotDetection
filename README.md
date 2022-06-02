# PKLotDetection repository

This is a repository for classifying parking lot images as empty or occupied. The associated technical report can be found in `report.pdf`.

## Overview of the repository

The repository contains several directories:

1. `code/`: this is the main code for running the training and get the results of the neural networks
2. `latex/`: directory containing the LaTeX source files as well as the `main.pdf` report
3. `pklotclass/`: Python library written and used in `code/` written using [PyTorch](https://pytorch.org) for the deep learning part.

## Structure of the `pklotclass/` library

The library is written in an object-oriented fashion. The content of each file is described below:

- `dataloader.py`: contains the `ParkingLotDataset` class which overloads the base `Dataset` class from PyTorch and reads parking lot datasets
- `eval.py`: performs evaluation of a neural network on a test dataset
- `log.py`: contains functions for creating logging objects
- `model.py`: contains the neural network architectures. A base class is first written which inherits from the `nn.Module` class of PyTorch. This base class holds a method for computing the number of training parameters and the other three classes inherit from it.
- `multiple_train.py`: script to launch multiple trainings sequentially
- `multiple_train.py`: script to launch multiple evaluations sequentially
- `pproc.py`: routines for post-processing the `metrics.h5` generated during training
- `predict.py`: script for inference of a given set of images
- `train.py`: performs the actual training of neural networks
- `trainer.py`: contains the `Trainer` class instantiated in `train.py`
- `util.py`: contains common functions used across the library

## Executables

The library is called via executables created in `setup.py`:

```python
entry_points={
    'console_scripts': [
        'pklot_train=pklotclass.train:main',
        'pklot_trains=pklotclass.multiple_train:main',
        'pklot_plot=pklotclass.pproc:main',
        'pklot_predict=pklotclass.predict:main'
    ],
},
```

Each of these executables are run via a configuration file written in `YAML` format (for example `code/results/200labels/simple.yml` for the `train_network` executable). These configuration files are converted into Python dictionnaries which are parsed throughout the code.

To install the library you need Python 3.8 or above version. Run the following command at the root of the repository (preferably in a virtual environment):

```shell
pip install -e .
```

If there are some missing packages, an image of a working python 3.8 version on Mac can be found in `requirements.txt` and so the following command should be run:

```shell
pip install -r requirements.txt
```

To run PyTorch on GPUs special versions of PyTorch should be installed with CUDA. Please refer to this [link](https://pytorch.org/get-started/previous-versions/) for a proper installation. For PyTorch 1.10.1 with CUDA 11.1 on Linux for example run the following command on terminal:

```shell
pip install torch==1.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```

## Training and evaluation

To reproduce the results shown in the technical report, first download the datasets in the root of the repository using the following shell script (you need `wget`):

```shell
./get_datasets.sh
```

Then for training and evaluation go to the `code/` repository and run the two shell scripts:

```shell
cd code/
./train.sh
./eval.sh
```

You can skip training and just evaluate the networks by downloading the pre-trained networks at this [link](https://mercure.cerfacs.fr/cfxfile/get.php?c926b48137b2) and then unzip inside the `code/` repository (note that you need a GPU with CUDA to relaunch the models in this case):

```shell
unzip train.zip
./eval.sh
```

Note that the models have been run on an NVIDIA A30 GPU.
