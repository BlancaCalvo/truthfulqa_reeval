#!/bin/bash
#SBATCH --job-name=train_judge
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4
#SBATCH --mem=20G
#SBATCH --time=24:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

export PATH=$PATH":/gaueko0/users/bcalvo/"
export PATH=$PATH":/gaueko0/users/bcalvo/.local/bin/"
export PATH=/usr/local/cuda-12.1/bin:$PATH

source ../truthfulqa-multi/new_env/bin/activate

# sh scripts/finetune.sh llama-3-truth-new meta-llama/Meta-Llama-3-8B data/finetune_truth.jsonl

# sh scripts/finetune.sh llama3.1.2 meta-llama/Meta-Llama-3.1-8B data/finetune_truth.jsonl

# sh scripts/finetune.sh llama3.1_multi meta-llama/Meta-Llama-3.1-8B data/finetune_truth_multi.jsonl

# sh scripts/finetune.sh gemma9b google/gemma-2-9b data/finetune_truth.jsonl

#sh scripts/finetune_instruct.sh llama3.1_instruct meta-llama/Meta-Llama-3.1-8B-Instruct data/finetune_truth.jsonl

#sh scripts/finetune.sh gemma9b_multi google/gemma-2-9b data/finetune_truth_multi.jsonl

# sh scripts/finetune.sh llama-3-multi meta-llama/Meta-Llama-3-8B data/finetune_truth_multi.jsonl

sh scripts/finetune_instruct.sh multi_gemma9b_instruct google/gemma-2-9b-it data/finetune_truth_multi.jsonl

# sh scripts/finetune_instruct.sh gemma9b_instruct google/gemma-2-9b-it data/finetune_truth.jsonl

#sh scripts/finetune_instruct.sh multi_llama3.1_instruct meta-llama/Meta-Llama-3.1-8B-Instruct data/finetune_truth_multi.jsonl