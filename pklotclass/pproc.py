# Post-processing metrics.h5 files from network trainings
# through use of a yaml config file
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import yaml
from pathlib import Path
from cycler import cycler

default_cycler = (cycler(color=['mediumblue', 'darkblue', 'firebrick', 'darkred']) +
         cycler(linestyle=['-', '--', '-', '--']))

plt.rc('axes', prop_cycle=default_cycler)

label_dict = {'accuracy': 'Accuracy',
        'loss': '$\mathcal{L}$'}

def ax_prop(ax, ylabel, logyscale):
    ax.set_xlabel('Epoch')
    ax.set_ylabel(ylabel)
    if logyscale: ax.set_yscale('log')
    ax.grid(True)
    ax.legend()

def main():
    # Parse cli argument
    args = argparse.ArgumentParser()
    args.add_argument('-c', '--config', type=str, required=True)
    args = args.parse_args()

    # convert yml file to dict
    with open(args.config, 'r') as yaml_stream:
        cfg = yaml.safe_load(yaml_stream)

    # Retrieve data from h5
    data_dir = Path(cfg['data_dir'])
    fig_dir = data_dir / 'figures' / cfg['fig_dir']
    fig_dir.mkdir(parents=True, exist_ok=True)
    data_networks = dict()

    if isinstance(cfg['type'], str):
        # If there is only one type (train or valid)
        for nn_name in cfg['networks']:
            data_networks[nn_name] = pd.read_hdf(data_dir / nn_name / 'metrics.h5', key=cfg['type'])

        # Plotting
        for figname in cfg['plot']:
            naxes = len(cfg['plot'][figname])
            fig, axes = plt.subplots(ncols=naxes, figsize=(5 * naxes, 4))
            if naxes == 1:
                axes = [axes]
            for i_qty, qty in enumerate(cfg['plot'][figname]):
                for nn_name, nn_data in data_networks.items():
                    axes[i_qty].plot(nn_data.index, nn_data[qty], label=cfg['networks'][nn_name])
                    ax_prop(axes[i_qty], label_dict[qty], qty=='loss')

            fig.savefig(fig_dir / f'{figname}.pdf', format='pdf', bbox_inches='tight')
    elif isinstance(cfg['type'], list):
        for nn_name in cfg['networks']:
            for dset_type in cfg['type']:
                data_networks[f'{cfg["networks"][nn_name]} {dset_type}'] = pd.read_hdf(data_dir / nn_name / 'metrics.h5', key=dset_type)
        # Plotting
        for figname in cfg['plot']:
            naxes = len(cfg['plot'][figname])
            fig, axes = plt.subplots(ncols=naxes, figsize=(5 * naxes, 4))
            if naxes == 1:
                axes = [axes]
            for i_qty, qty in enumerate(cfg['plot'][figname]):
                for nn_name, nn_data in data_networks.items():
                    axes[i_qty].plot(nn_data.index, nn_data[qty], label=nn_name)
                    ax_prop(axes[i_qty], label_dict[qty], qty=='loss')

            fig.savefig(fig_dir / f'{figname}.pdf', format='pdf', bbox_inches='tight')

if __name__ == '__main__':
    main()