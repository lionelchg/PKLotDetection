#!/bin/sh

# Create dataset directory
mkdir datasets/
cd datasets/ || exit

# 1. CNRPark dataset
wget "http://cnrpark.it/dataset/CNRPark-Patches-150x150.zip" -O "CNRPark.zip"
mkdir CNRPark-Patches-150x150
unzip CNRPark.zip -d CNRPark-Patches-150x150 && rm CNRPark.zip

# 2. splits with labels
wget "http://cnrpark.it/dataset/splits.zip" -O "splits.zip"
unzip splits.zip && rm splits.zip