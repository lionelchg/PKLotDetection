##################################################################################################################
#                                                                                                                #
#                                 pklotclass.multiple_eval: eval multiple models                               #
#                                                                                                                #
#                                          Lionel Cheng, 01.06.2022                                              #
#                                                                                                                #
##################################################################################################################

import os
import argparse
import re
import yaml
import copy
from pathlib import Path
from itertools import product

import numpy as np
import torch

from .eval import run_eval
from .multiple_train import generate_train_cfgs, set_nested

def run_evals(networks, base_cfg):
    """ Create the configuration files for each run and yield it to be read by
    each run function """
    for ncase, network in networks.items():
        # deepcopy is very important for recursive copy
        case_cfg = copy.deepcopy(base_cfg)
        for key, value in network.items():
            n_slash = key.count("/")
            nstr = n_slash + 1
            str_re = r'(\w*)/' * n_slash + r'(\w*)'
            re_keys = re.search(str_re, key)
            keys_tmp = []
            for i in range(nstr):
                keys_tmp.append(re_keys.group(i + 1))
            set_nested(case_cfg, value, *keys_tmp)

        # Run evaling
        run_eval(case_cfg)

def main():
    args = argparse.ArgumentParser(description='PlasmaNet.nnet')
    args.add_argument('-c', '--config', required=True, type=str,
                        help='Config file path (default: None)')
    args = args.parse_args()

    with open(args.config, 'r') as yaml_stream:
        cfg_networks = yaml.safe_load(yaml_stream)

    # Parse main config file
    networks, base_cfg = generate_train_cfgs(cfg_networks)

    # Run the different wanted configurations
    run_evals(networks, base_cfg)

if __name__ == '__main__':
    main()