# Code for MASS

This repository contains the official implementation of the paper "MASS: Modality-collaborative semi-supervised segmentation by exploiting cross-modal consistency from unpaired CT and MRI images".

## Dataset
Our experiments were conducted on the combination of two existing datasets: [BTCV](https://www.synapse.org/#!Synapse:syn3193805/wiki/) and [CHAOS](https://chaos.grand-challenge.org/). The combined dataset comprises 47 CT images (from BTCV) and 40 MR images (CHAOS). 

To speed up data reading, we first pre-process each image using offline scaling and save it locally. In practice, you can use the script in ``Code/BTCV_preprocess/data_process.ipynb`` to pre-process the BTCV dataset, and scripts in ``Code/CHAOS_preprocess/process_label.ipynb`` and ``Code/CHAOS_preprocess/process_nii.ipynb`` to pre-process the CHAOS dataset.

## Train
This is the command for using one GPU for training.

```
python train.py --path path_to_combined_dataset --save_path save_path --epochs N --batch_size B
```

## Test
After finishing training, the model will be saved to ``save_path``. Please use the following command for testing.
```
python test.py  
```
