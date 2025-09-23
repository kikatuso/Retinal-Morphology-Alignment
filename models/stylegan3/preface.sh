#!/bin/bash


## FIRST CHECK module avail CUDA as depends what GPUs node you are going to get
### Pick the default module (its the one pytorch uses)
## if this does not work, you may need to remove old plugins
## then remove old plugins rm -rf /well/papiez/users/zwk579/.cache/torch_extensions/bias_act_plugin
## rm -rf /gpfs3/well/papiez/users/zwk579/.cache/torch_extensions/py311_cu126


module load CUDA/12.6.0
#module load CUDA/12.1.1

module load GCC/12.3.0

# # Activate your conda environment
source /apps/eb/el8/common/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate /gpfs3/well/papiez/users/zwk579/.conda_envs/myenv 

# # Set Python include paths for compiler
# # Set Python include paths for compiler
export CPATH=/gpfs3/well/papiez/users/zwk579/.conda_envs/myenv/include/python3.11:$CPATH
export CPLUS_INCLUDE_PATH=/gpfs3/well/papiez/users/zwk579/.conda_envs/myenv/include/python3.11:$CPLUS_INCLUDE_PATH


echo "Python include paths set to:"
echo "CPATH: $CPATH"
echo "CPLUS_INCLUDE_PATH: $CPLUS_INCLUDE_PATH"

echo $CUDA_HOME



