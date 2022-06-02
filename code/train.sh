#!/bin/bash

# Run training for mAlexNet, AlexNet and LeNet5
pklot_trains -c train_networks.yml

# Plot metrics
pklot_plot -c pproc.yml