#!/bin/bash
DATASET_DIR="data"
WORKSPACE="."
# Pack csv files to hdf5
#python3 pack_csv_to_hdf5.py --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE
# sleep 2h
# Train
# python3 main_pytorch.py train --config $WORKSPACE/example_config.json --workspace=$WORKSPACE --cuda
# python3 main_pytorch.py train --config $WORKSPACE/example_config_kettle.json --workspace=$WORKSPACE --cuda
# python3 main_pytorch.py train --config $WORKSPACE/example.json --workspace=$WORKSPACE --cuda
# python3 main_pytorch.py train --config $WORKSPACE/example_config_washingmachine_bert.json --workspace=$WORKSPACE --cuda
# python3 main_pytorch.py train --config $WORKSPACE/example_config_washingmachine_bert_finetune.json --workspace=$WORKSPACE --cuda
# python3 main_pytorch.py train --config $WORKSPACE/example_config_kettle_bert.json --workspace=$WORKSPACE --cuda
# python3 main_pytorch.py train --config $WORKSPACE/example_config_kettle_bert_finetune.json --workspace=$WORKSPACE --cuda
# python3 main_pytorch.py train --config $WORKSPACE/example_config_dishwasher_bert.json --workspace=$WORKSPACE --cuda
# python3 main_pytorch.py train --config $WORKSPACE/example_config_dishwasher_bert_finetune.json --workspace=$WORKSPACE --cuda
# python3 main_pytorch.py train --config $WORKSPACE/example_config_microwave_bert_finetune.json --workspace=$WORKSPACE --cuda

# Inference
# python3 main_pytorch.py inference --config $WORKSPACE/example_config_kettle.json --workspace=$WORKSPACE  --cuda
# python3 main_pytorch.py inference --config $WORKSPACE/example_config_dishwasher_bert_finetune.json --workspace=$WORKSPACE --batch_size 1  --cuda
# python3 main_pytorch.py inference --config $WORKSPACE/example_config_dishwasher_bert.json --workspace=$WORKSPACE  --cuda
# python3 main_pytorch.py inference --config $WORKSPACE/example_config_washingmachine_bert.json --workspace=$WORKSPACE --batch_size 1  --cuda
# python3 main_pytorch.py inference --config $WORKSPACE/example_config_washingmachine_bert_finetune.json --workspace=$WORKSPACE --batch_size 1  --cuda
# python3 main_pytorch.py inference --config $WORKSPACE/example_config_kettle_bert.json --workspace=$WORKSPACE --batch_size 1  --cuda
# python3 main_pytorch.py inference --config $WORKSPACE/example_config_kettle_bert_finetune.json --workspace=$WORKSPACE --batch_size 1  --cuda
python3 main_pytorch.py inference --config $WORKSPACE/example_config_microwave_bert_finetune.json --workspace=$WORKSPACE --batch_size 1  --cuda
# python3 main_pytorch.py inference --config $WORKSPACE/example.json --workspace=$WORKSPACE  --cuda
# python3 main_pytorch.py inference --config $WORKSPACE/example_config.json --workspace=$WORKSPACE  --cuda
