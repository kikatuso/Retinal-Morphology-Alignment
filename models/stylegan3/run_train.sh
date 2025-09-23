#!/bin/bash
source ./preface.sh


#folder='/stylegan3/log/stylegan3-ukb/00022-stylegan3-t-256x256px-gpus2-batch32-gamma10'
folder=''
resume_path=$(ls "$folder"/network-snapshot-*.pkl 2>/dev/null | sort -V | tail -n 1)

resume_kimg=$(echo "$resume_path" | sed -n 's/.*-\([0-9]\{6\}\)\.pkl/\1/p' | sed 's/^0*//')
# Fallback to 0 if empty
if [ -z "$resume_kimg" ]; then
    resume_kimg=0
fi

echo 'starting at epoch' $resume_kimg 'with path from' $resume_path

python train.py \
  --outdir /stylegan3/log \
  --modelname stylegan3-ukb \
  --cfg stylegan3-t \
  --data /retinal_data/256x256px \
  --gpus 1 \
  --batch 32 \
  --gamma 10.0 \
  --kimg 3001 \
  --snap 10 \
  --resume $resume_path \
  --resume_kimg $resume_kimg \
  --metrics none 