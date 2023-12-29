#!/bin/bash
source activate anomalib_env
datapath=./data/MVTec
loadpath=./models

modelfolder=IM320_Ensemble_L2-3_P001_D1024-384_PS-3_AN-1
# modelfolder=IM224_Ensemble_L2-3_P001_D1024-384_PS-3_AN-1
savefolder=evaluated_results'/'$modelfolder

datasets=('bottle'  'cable'  'capsule'  'carpet'  'grid'  'hazelnut' 'leather'  'metal_nut'  'pill' 'screw' 'tile' 'toothbrush' 'transistor' 'wood' 'zipper')
model_flags=($(for dataset in "${datasets[@]}"; do echo '-p '$loadpath'/'$modelfolder'/models/mvtec_'$dataset; done))
dataset_flags=($(for dataset in "${datasets[@]}"; do echo '-d '$dataset; done))

python bin/load_and_evaluate_patchcore.py --gpu 0 --save_segmentation_images --seed 0 evaluated_results/capsule patch_core_loader -p ./models/IM320_WR50_L2-3_P001_D1024-1024_PS-5_AN-3/models/mvtec_capsule --faiss_on_gpu dataset --resize 366 --imagesize 320 -d capsule mvtec ./data/MVTec
bash