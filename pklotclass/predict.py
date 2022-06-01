############################################################################################
#                                                                                          #
#                                   Main prediction routine                                #
#                                                                                          #
#                                   Lionel Cheng, 01.06.2022                               #
#                                                                                          #
############################################################################################
# PyTorch
import torch
from torch.utils.data import Dataset
from torchvision import transforms

# Others
from sklearn.preprocessing import LabelEncoder
import yaml
import argparse
import numpy as np
from pathlib import Path
import os
from PIL import Image

# Internal routines
import pklotclass.model as pkmodel
from .log import create_log

family_dict = {0: 'Empty', 1: 'Occupied'}

class Predictor:
    """ Class for inference of given input sequences. """
    def __init__(self, model, img_dataset, cfg):
        # Create logger for training
        self.cfg = cfg
        self.save_dir = Path(self.cfg['location']) / 'eval'
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.logger = create_log('predict', self.save_dir, logformat='small', console=True)

        # Copy configuration dictionnary in case folder
        with open(self.save_dir / 'config.yml', 'w') as file:
            yaml.dump(cfg, file)

        # Store the sequences dataset
        self.img_dataset = img_dataset

        # setup GPU device if available, move model into configured device
        self.device, device_ids = self._prepare_device(self.cfg['n_gpu'])
        self.model = model.to(self.device)
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        # Resume training
        self._resume_checkpoint(self.cfg['location'] + '/model_best.pth')

    def predict(self):
        """ Predict the label of a list of sequences. """
        for idx, img in enumerate(self.img_dataset):
            img = img.to(self.device)
            predicted_label_vec = self.model(img)
            label = predicted_label_vec.argmax(1)
            family = family_dict[label]
            self.logger.info(f'Image #{idx+1:d} belongs to {family} family')

    def _prepare_device(self, n_gpu_use):
        """ Setup GPU device if available, move model into configured device. """
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            self.logger.warning("Warning: There\'s no GPU available on this machine,"
                                "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            self.logger.warning("Warning: The number of GPU\'s configured to use is {}, but only {} are available "
                                "on this machine.".format(n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _resume_checkpoint(self, resume_path):
        """ Resume from the given saved checkpoint. """
        resume_path = str(resume_path)
        self.logger.info('Loading checkpoint: {} ...'.format(resume_path))

        checkpoint = torch.load(resume_path)
        self.model.load_state_dict(checkpoint['state_dict'])

        self.logger.info('Checkpoint loaded.')


class ParkingLotDataset(Dataset):
    def __init__(self, img_dir, img_path, transforms = None):
        with open(img_path, 'r') as f:
            lines = f.readlines()
            self.img_list = [os.path.join(img_dir, i.split()[0]) for i in lines]
            self.transforms = transforms

    def __getitem__(self, index):
        try:
            img_path = self.img_list[index]
            img = Image.open(img_path)
            img = self.transforms(img)
        except:
            return None
        return img

    def __len__(self):
        return len(self.label_list)


def run_predict(cfg):
    """ Run training based on configuration dictionnary. """
    # First the model
    model_cfg = cfg['model']
    model = getattr(pkmodel, model_cfg['type'])(**model_cfg['args'])
    # Hardcoded transformation as in Amato 2017
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),  # normalize to [0, 1]
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Creation of dataset from the sequence list
    img_dir = cfg['img_dir']
    img_path = cfg['img_path']
    img_dataset = ParkingLotDataset(img_dir, img_path, transform)

    # Creation of trainer object
    predictor = Predictor(model, img_dataset, cfg)

    # Training step
    predictor.predict()

def main():
    """ Wrapper around training routine.
    Parse the configuration dictionnary. """
    # Open configuration dictionnary
    parser = argparse.ArgumentParser(
        description='Training of Protein classification model')
    parser.add_argument('-c', '--config', type=str, required=True)
    args = parser.parse_args()
    with open(args.config, 'r') as yaml_stream:
        cfg = yaml.safe_load(yaml_stream)

    # Run prediction
    run_predict(cfg)

if __name__ == '__main__':
    main()