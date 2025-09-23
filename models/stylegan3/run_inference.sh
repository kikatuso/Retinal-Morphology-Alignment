#!/bin/bash
source ./preface.sh


HOME=/gpfs3/well/papiez/users/zwk579

echo "saving model to"  $HOME/.cache/dnnlib/



## pictures of dogs - nvidia model
# python3.11 gen_images.py \
#   --modelname stylegan3-r-afhqv2-512x512 \
#   --trunc=1 \
#   --seeds=64,32,23,1 \
#   --network=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-afhqv2-512x512.pkl
  
if [ "$module_name" = "stylegan3-medfusion" ]; then


python gen_images.py \
  --modelname stylegan3-medfusion \
  --trunc=1.0 \
  --seeds $seed_str \
  --network /gpfs3/well/papiez/users/zwk579/Results/stylegan3/log/stylegan3-medfusion/saved_model.pkl


elif [ "$module_name" = "stylegan3-ukb" ]; then

### our trained model
python gen_images.py \
  --modelname stylegan3-ukb \
  --trunc=1.0 \
  --seeds $seed_str \
  --network /gpfs3/well/papiez/users/zwk579/Results/stylegan3/log/stylegan3-ukb/00022-stylegan3-t-256x256px-gpus2-batch32-gamma10/network-snapshot-003000.pkl

else
    echo "Error: Unknown module_name '$module_name'"
    exit 1
fi