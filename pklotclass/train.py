############################################################################################
#                                                                                          #
#                                    Main training routine                                 #
#                                                                                          #
#                                   Lionel Cheng, 01.06.2022                               #
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

def run_train(cfg):
    """ Run training based on configuration dictionnary. """
    # Parse the configuration dicionnary
    train_img_path = cfg['train_img_path']
    train_img_labels = cfg['train_img_labels']
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
    train_dataset = ParkingLotDataset(train_img_path, train_img_labels, transform)
    if 'val_img_labels' in cfg:
        val_img_path = cfg['val_img_path']
        val_img_labels = cfg['val_img_labels']
        val_dataset = ParkingLotDataset(val_img_path, val_img_labels, transform)
    else:
        len_val = int(0.2 * len(train_dataset))
        len_train = len(train_dataset) - len_val
        train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, (len_train, len_val))
    test_dataset = ParkingLotDataset(test_img_path, test_img_labels, transform)

    # Create dataloader objects
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # Creation of the training related objects
    # First the model
    model_cfg = cfg['model']
    model = getattr(pkmodel, model_cfg['type'])(**model_cfg['args'])

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = getattr(torch.optim, cfg['optimizer']['type'])(
        model.parameters(), **cfg['optimizer']['args'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **cfg['scheduler'])

    # Creation of trainer object
    trainer = Trainer(model, criterion, optimizer,
                train_dataloader, val_dataloader, test_dataloader,
                scheduler, cfg)

    # Training step
    trainer.train()

    # Evaluate the model on the test dataset at the end of training
    trainer.test()

def main():
    """ Wrapper around training routine.
    Parse the configuration dictionnary. """
    # Open configuration dictionnary
    parser = argparse.ArgumentParser(
        description='Training of parking lot classification model')
    parser.add_argument('-c', '--config', type=str, required=True)
    args = parser.parse_args()
    with open(args.config, 'r') as yaml_stream:
        cfg = yaml.safe_load(yaml_stream)

    # Run training
    run_train(cfg)

if __name__ == '__main__':
    main()