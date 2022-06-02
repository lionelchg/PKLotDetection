############################################################################################
#                                                                                          #
#                                  Main evaluation routine                                 #
#                                                                                          #
#                                 Lionel Cheng, 01.06.2022                                 #
#                                                                                          #
############################################################################################
# PyTorch
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

# Others
import yaml
import argparse

# Internal routines
import pklotclass.model as pkmodel
from .dataloader import ParkingLotDataset
from .trainer import Trainer

def run_eval(cfg):
    """ Run training based on configuration dictionnary. """
    # Parse the configuration dicionnary
    test_img_path = cfg['test_img_path']
    test_img_labels = cfg['test_img_labels']
    batch_size = cfg['batch_size']

    # Hardcoded transformation as in Amato 2017
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),  # normalize to [0, 1]
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Create dataset objects
    test_dataset = ParkingLotDataset(test_img_path, test_img_labels, transform)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # Creation of the training related objects
    # First the model
    model_cfg = cfg['model']
    model = getattr(pkmodel, model_cfg['type'])(**model_cfg['args'])

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 2, gamma=0.1)

    # Creation of trainer object
    trainer = Trainer(model, criterion, optimizer,
                test_dataloader, test_dataloader, test_dataloader,
                scheduler, cfg)

    # Evaluate the model on the test dataset at the end of training
    trainer.test()

def main():
    """ Wrapper around evaluation routine.
    Parse the configuration dictionnary. """
    # Open configuration dictionnary
    parser = argparse.ArgumentParser(
        description='Training of parking lot classification model')
    parser.add_argument('-c', '--config', type=str, required=True)
    args = parser.parse_args()
    with open(args.config, 'r') as yaml_stream:
        cfg = yaml.safe_load(yaml_stream)

    # Run training
    run_eval(cfg)

if __name__ == '__main__':
    main()