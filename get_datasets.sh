#!/bin/sh

# Create dataset directory
mkdir datasets/
cd datasets/ || exit

# 1. CNRPark dataset
wget "http://cnrpark.it/dataset/CNRPark-Patches-150x150.zip" -O "CNRPark.zip"
mkdir CNRPark-Patches-150x150
unzip CNRPark.zip -d CNRPark-Patches-150x150 && rm CNRPark.zip

# 2. CNRPark-EXT dataset
wget "http://cnrpark.it/dataset/CNR-EXT-Patches-150x150.zip" -O "CNRPark_EXT.zip"
unzip CNRPark_EXT.zip && rm CNRPark_EXT.zip

# 3. PKLot dataset
wget "http://www.inf.ufpr.br/vri/databases/PKLot.tar.gz" -O "PKLot.tar.gz"
tar -xvf PKLot.tar.gz && rm PKLot.tar.gz

# 4. splits with labels
wget "http://cnrpark.it/dataset/splits.zip" -O "splits.zip"
unzip splits.zip && rm splits.zip