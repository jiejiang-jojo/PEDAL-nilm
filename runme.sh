#!/bin/bash
DATASET_DIR="data"
WORKSPACE="."
# Pack csv files to hdf5
#python3 pack_csv_to_hdf5.py --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE
# Train
# python3 main_pytorch.py train --config $WORKSPACE/example_config.json --workspace=$WORKSPACE --cuda

# Inference
python3 main_pytorch.py inference --config $WORKSPACE/example_config.json --workspace=$WORKSPACE --batch_size 1  --cuda
