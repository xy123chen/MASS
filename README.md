# Code for MASS


This repository provides the official implementation of our paper " MASS: Modality-collaborative semi-supervised segmentation  by exploiting cross-modal consistency from unpaired CT and MRI images".

## Dataset
We build a publicly available multi-modal multiorgan abdominal . The dataset comprises 47 CT images (from [BTCV](https://www.synapse.org/#!Synapse:syn3193805/wiki/)) and 40 MR images (from [CHAOS](https://chaos.grand-challenge.org/)). 

In order to facilitate the reading of the data, we first preprocess the image using offline scaling and save it locally. You can use Code/BTCV_preprocess/data_process.ipynb to preprocessing the BTCV dataset and use Code/CHAOS_preprocess/process_label.ipynb, Code/CHAOS_preprocess/process_nii.ipynb to preprocessing the CHAOS.

## Train
This is the command for using 1 GPU for training.

```
CUDA_VISIBLE_DEVICES=5 python train.py --path ../Dataset/CT_MR/ --save_path ../Weight/  --epochs 5000 --batch_size 1
```

## Test
After the training, the model will be saved in the /Weight. Use the following command for testing.
```
python test.py  
```
