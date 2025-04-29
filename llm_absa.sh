#!/bin/bash

#SBATCH --partition=hard
#SBATCH --job-name=llama_beauty
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/llama_beauty.out
#SBATCH --error=logs/llama_beauty.err

python llm_absa.py\
    --model /home/kabongo/llama/llama3/Meta-Llama-3-8B-Instruct \
    --domain beauty \
    --dataset_path /data/common/RecommendationDatasets/Beauty_Amazon14/reviews.json \
    --output_dir /data/common/RecommendationDatasets/Beauty_Amazon14/ \
    --batch_size 32 \
    --max_new_tokens 512