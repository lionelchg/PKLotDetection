############################################################################################
#                                                                                          #
#                                        Trainer class                                     #
#                                                                                          #
#                                   Lionel Cheng, 01.06.2022                               #
#                                                                                          #
############################################################################################
import torch
import time
from numpy import inf
import numpy as np
from pathlib import Path
import pandas as pd
import yaml

from .log import create_log
from .util import MetricTracker

def conditioned_stats(TP, TN, FP, FN):
    try:
        TPR = TP / (TP + FN)
    except ZeroDivisionError:
        TPR = -1
    try:
        TNR = TN / (TN + FP)
    except ZeroDivisionError:
        TNR = -1
    try:
        PPV = TP / (TP + FP)
    except ZeroDivisionError:
        PPV = -1
    try:
        NPV = TN / (TN + FN)
    except ZeroDivisionError:
        NPV = -1
    return TPR, TNR, PPV, NPV

class Trainer:
    def __init__(self, model, criterion, optimizer,
                train_dataloader, valid_dataloader, test_dataloader,
                scheduler, cfg):
        """ Initialize Trainer object. It needs a model (the neural network),
        a criterion (the loss), an optimizer for the training procedure.
        The training, validation and test data loaders are also given.
        The encoder of the labels is also necessary for storage.
        """
        # Objects declaration for training
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.test_dataloader = test_dataloader
        self.scheduler = scheduler

        # Create logger for training
        self.cfg_global = cfg
        self.cfg = cfg['trainer']
        self.save_dir = Path(self.cfg['save_dir'])
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.logger = create_log('train', self.save_dir, logformat='small', console=False)

        # Copy configuration dictionnary in case folder
        with open(self.save_dir / 'config.yml', 'w') as file:
            yaml.dump(cfg, file)

        # setup GPU device if available, move model into configured device
        self.device, device_ids = self._prepare_device(self.cfg['n_gpu'])
        self.model = model.to(self.device)
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        # Hold metrics
        self.train_metrics = MetricTracker('loss', 'accuracy', 'TPR', 'TNR', 'PPV', 'NPV')
        self.valid_metrics = MetricTracker('loss', 'accuracy', 'TPR', 'TNR', 'PPV', 'NPV')
        self.df_train_metrics = pd.DataFrame(columns=('loss', 'accuracy', 'TPR', 'TNR', 'PPV', 'NPV'))
        self.df_valid_metrics = pd.DataFrame(columns=('loss', 'accuracy', 'TPR', 'TNR', 'PPV', 'NPV'))

        # Small parameters setting
        self.epochs = self.cfg['epochs']
        self.save_period = self.cfg['save_period']
        self.log_interval = int(10**int(np.log10(len(self.train_dataloader))))
        self.start_epoch = 1

        # Configuration to monitor model performance and save best
        self.monitor = self.cfg['monitor']
        self.mnt_mode, self.mnt_metric = self.monitor.split()
        assert self.mnt_mode in ['min', 'max']

        self.mnt_best = inf if self.mnt_mode == 'min' else -inf
        self.early_stop = self.cfg.get('early_stop', inf)

        # Resume training
        if 'resume' in self.cfg:
            self._resume_checkpoint(self.cfg['resume'])

    def _train_epoch(self, epoch):
        """ Training method for a given epoch. """
        self.model.train()
        total_acc, total_count = 0, 0

        # Print headers
        self.logger.info('-' * 110)
        self.logger.info(f'| {"Epoch":>10} | {"Batches":>13} | {"Accuracy":>10} '
            f'| {"TPR":>10} | {"TNR":>10} | {"PPV":>10} | {"NPV":>10} | {"Loss":>12} |')
        for idx, (batch_images, batch_labels) in enumerate(self.train_dataloader):
            # Convert the labels to right format
            batch_labels = list(map(int, batch_labels))
            batch_labels = torch.Tensor(batch_labels)

            # Transfer to data to device (necessary for GPU)
            batch_images = batch_images.to(self.device)
            batch_labels = batch_labels.to(self.device)

            # Prediction of the network
            self.optimizer.zero_grad()
            predicted_labels = self.model(batch_images)

            # Compute the loss and backpropagate
            loss = self.criterion(predicted_labels, batch_labels.long())
            loss.backward()
            self.optimizer.step()

            # Compute the accuracy
            local_acc = (predicted_labels.argmax(1) == batch_labels).sum().item()
            total_acc += local_acc
            local_count = batch_labels.size(0)
            total_count += local_count

            # Compute conditioned statistics
            TP = ((predicted_labels.argmax(1) == 1).cpu().detach().numpy() & (batch_labels == 1).cpu().detach().numpy()).sum().item()
            TN = ((predicted_labels.argmax(1) == 0).cpu().detach().numpy() & (batch_labels == 0).cpu().detach().numpy()).sum().item()
            FP = ((predicted_labels.argmax(1) == 1).cpu().detach().numpy() & (batch_labels == 0).cpu().detach().numpy()).sum().item()
            FN = ((predicted_labels.argmax(1) == 0).cpu().detach().numpy() & (batch_labels == 1).cpu().detach().numpy()).sum().item()
            TPR, TNR, PPV, NPV = conditioned_stats(TP, TN, FP, FN)

            # Print info
            if idx % self.log_interval == 0 and idx > 0:
                self.logger.info('| {:10d} | {:6d}/{:6d} '
                    '| {:10.3f} | {:10.3f} | {:10.3f} | {:10.3f} | {:10.3f} | {:12.4e} |'.format(epoch, idx, len(self.train_dataloader),
                                                total_acc / total_count, TPR, TNR, PPV, NPV, loss))
                total_acc, total_count = 0, 0

            # Update metric trackers
            batch_len = len(batch_labels)
            self.train_metrics.update('loss', loss.item(), batch_len)
            self.train_metrics.update('accuracy', local_acc / local_count, batch_len)
            self.train_metrics.update('TPR', TPR, batch_len)
            self.train_metrics.update('TNR', TNR, batch_len)
            self.train_metrics.update('PPV', PPV, batch_len)
            self.train_metrics.update('NPV', PPV, batch_len)

        self.df_train_metrics.loc[epoch] = self.train_metrics._data.average


    def _valid_epoch(self, epoch):
        """ Valid method for a given epoch. """
        self.model.eval()
        total_acc, total_count = 0, 0
        total_TP, total_TN, total_FP, total_FN = 0, 0, 0, 0

        with torch.no_grad():
            for idx, (batch_images, batch_labels) in enumerate(self.valid_dataloader):
                # Convert the labels to right format
                batch_labels = list(map(int, batch_labels))
                batch_labels = torch.Tensor(batch_labels)

                # Transfer to data to device (necessary for GPU)
                batch_images = batch_images.to(self.device)
                batch_labels = batch_labels.to(self.device)

                # Prediction of the network
                predicted_labels = self.model(batch_images)

                # Compute the loss
                loss = self.criterion(predicted_labels, batch_labels.long())

                # Compute the accuracy
                local_acc = (predicted_labels.argmax(1) == batch_labels).sum().item()
                total_acc += local_acc
                local_count = batch_labels.size(0)
                total_count += local_count

                # Compute conditioned statistics
                TP = ((predicted_labels.argmax(1) == 1).cpu().detach().numpy() & (batch_labels == 1).cpu().detach().numpy()).sum().item()
                TN = ((predicted_labels.argmax(1) == 0).cpu().detach().numpy() & (batch_labels == 0).cpu().detach().numpy()).sum().item()
                FP = ((predicted_labels.argmax(1) == 1).cpu().detach().numpy() & (batch_labels == 0).cpu().detach().numpy()).sum().item()
                FN = ((predicted_labels.argmax(1) == 0).cpu().detach().numpy() & (batch_labels == 1).cpu().detach().numpy()).sum().item()
                TPR, TNR, PPV, NPV = conditioned_stats(TP, TN, FP, FN)
                total_TP += TP
                total_TN += TN
                total_FP += FP
                total_FN += FN

                # Update metric trackers
                batch_len = len(batch_labels)
                self.valid_metrics.update('loss', loss.item(), batch_len)
                self.valid_metrics.update('accuracy', local_acc / local_count, batch_len)
                self.valid_metrics.update('TPR', TPR, batch_len)
                self.valid_metrics.update('TNR', TNR, batch_len)
                self.valid_metrics.update('PPV', PPV, batch_len)
                self.valid_metrics.update('NPV', NPV, batch_len)

            self.df_valid_metrics.loc[epoch] = self.valid_metrics._data.average
        TPR, TNR, PPV, NPV = conditioned_stats(total_TP, total_TN, total_FP, total_FN)
        return total_acc / total_count, TPR, TNR, PPV, NPV

    def train(self):
        """ Full training method. """
        # Print the model
        self.logger.info('Model:')
        self.logger.info(self.model)
        self.logger.info('Optimizer:')
        self.logger.info(self.optimizer)
        total_accu = None
        for epoch in range(1, self.epochs + 1):
            epoch_start_time = time.time()
            self._train_epoch(epoch)
            accu_val, TPR, TNR, PPV, NPV = self._valid_epoch(epoch)
            if total_accu is not None and total_accu > accu_val:
                # If accuracy stops to improve we decrease the learning rate
                self.scheduler.step()
                self.logger.info('| Accuracy stops improving, new value: '
                    f'{self.optimizer.param_groups[0]["lr"]:.2e}')
            else:
                # Accuracy has improved so we save the model
                total_accu = accu_val
                self._save_best(epoch)
            self.logger.info('-' * 110)
            self.logger.info('| End of epoch {:3d} | time: {:5.2f}s | '.format(epoch,
                                                time.time() - epoch_start_time))
            self.logger.info(f'| {"Accuracy":>10} '
                f'| {"TPR":>10} | {"TNR":>10} | {"PPV":>10} | {"NPV":>10} |')
            self.logger.info('| {:10.3f} | {:10.3f} | {:10.3f} | {:10.3f} | {:10.3f} |'.format(
                        accu_val, TPR, TNR, PPV, NPV))
            self.logger.info('-' * 110 + '\n')

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch)

            # Save metrics.h5
            self.df_train_metrics.to_hdf(self.save_dir / 'metrics.h5', key='train', mode='w')
            self.df_valid_metrics.to_hdf(self.save_dir / 'metrics.h5', key='valid', mode='a')

    def test(self):
        """ Run the model on the test dataset. """
        self.model.eval()
        total_acc, total_count = 0, 0
        total_TP, total_TN, total_FP, total_FN = 0, 0, 0, 0

        with torch.no_grad():
            for _, (batch_images, batch_labels) in enumerate(self.test_dataloader):
                # Convert the labels to right format
                batch_labels = list(map(int, batch_labels))
                batch_labels = torch.Tensor(batch_labels)

                # Transfer to data to device (necessary for GPU)
                batch_images = batch_images.to(self.device)
                batch_labels = batch_labels.to(self.device)

                # Prediction of the network
                predicted_labels = self.model(batch_images)

                # Compute the accuracy
                local_acc = (predicted_labels.argmax(1) == batch_labels).sum().item()
                total_acc += local_acc
                local_count = batch_labels.size(0)
                total_count += local_count

                # Compute conditioned statistics
                TP = ((predicted_labels.argmax(1) == 1).cpu().detach().numpy() & (batch_labels == 1).cpu().detach().numpy()).sum().item()
                TN = ((predicted_labels.argmax(1) == 0).cpu().detach().numpy() & (batch_labels == 0).cpu().detach().numpy()).sum().item()
                FP = ((predicted_labels.argmax(1) == 1).cpu().detach().numpy() & (batch_labels == 0).cpu().detach().numpy()).sum().item()
                FN = ((predicted_labels.argmax(1) == 0).cpu().detach().numpy() & (batch_labels == 1).cpu().detach().numpy()).sum().item()
                total_TP += TP
                total_TN += TN
                total_FP += FP
                total_FN += FN

            TPR, TNR, PPV, NPV = conditioned_stats(total_TP, total_TN, total_FP, total_FN)
            self.logger.info('-' * 110)
            self.logger.info('| Evaluating network on test dataset. Size: {:6d} '.format(len(self.test_dataloader)))
            self.logger.info(f'| {"Accuracy":>10} '
                f'| {"TPR":>10} | {"TNR":>10} | {"PPV":>10} | {"NPV":>10} |')
            self.logger.info('| {:10.3f} | {:10.3f} | {:10.3f} | {:10.3f} | {:10.3f} |'.format(
                        total_acc / total_count, TPR, TNR, PPV, NPV))
            self.logger.info('-' * 110)

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

    def _save_checkpoint(self, epoch):
        """ Saving checkpoints. """
        # Create checkpoint dict to be saved
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best,
        }
        filename = str(self.save_dir / 'checkpoint-epoch{:05d}.pth'.format(epoch))
        # Save checkpoint
        torch.save(state, filename)
        self.logger.info('Saving checkpoint: {} ...'.format(filename))

    def _save_best(self, epoch):
        """ Saving the best model """
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best,
        }
        best_path = str(self.save_dir / 'model_best.pth')
        torch.save(state, best_path)
        self.logger.info('Saving current best: model_best.pth ...')

    def _resume_checkpoint(self, resume_path):
        """ Resume from the given saved checkpoint. """
        resume_path = str(resume_path)
        self.logger.info('Loading checkpoint: {} ...'.format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']
        # Load architecture params from checkpoint
        if checkpoint['arch'] != self.cfg_global['model']['type']:
            self.logger.warning('Warning: Architecture configuration from the config file and the checkpoint is '
                                'different. This may yield an exception while state_dict is loaded.')
        self.model.load_state_dict(checkpoint['state_dict'])

        # Load optimizer state from checkpoint only if optimizer type is not changed
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info('Checkpoint loaded. Resume training from epoch {}'.format(self.start_epoch))

