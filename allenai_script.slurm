#!/bin/bash
#SBATCH --job-name=train_allenai
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4
#SBATCH --mem=20G
#SBATCH --time=24:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

export PATH=$PATH":/gaueko0/users/bcalvo/"
export PATH=$PATH":/gaueko0/users/bcalvo/.local/bin/"
export PATH=/usr/local/cuda-12.1/bin:$PATH

source ../git_truthfulqa-multi/env/bin/activate

%HF_HOME='/ixadata/users/bcalvo/cache'

sh scripts/finetune_judge.sh