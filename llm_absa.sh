#!/bin/bash

#SBATCH --partition=electronic
#SBATCH --job-name=llama_beauty
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=3000
#SBATCH --output=logs/llama_beauty.out
#SBATCH --error=logs/llama_beauty.err

python llm_absa.py\
    --model /home/kabongo/llama/llama3/Meta-Llama-3-8B-Instruct \
    --domain movies \
    --dataset_path /data/common/RecommendationDatasets/Beauty_Amazon/reviews_filtered.jsonl \
    --output_dir /data/common/RecommendationDatasets/Beauty_Amazon/v0/ \
    --json_format \
    --batch_size 32 \
    --max_new_tokens 512 \
    --skip_existing \