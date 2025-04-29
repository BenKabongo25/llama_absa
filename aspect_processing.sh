#!/bin/bash

#SBATCH --partition=hard
#SBATCH --job-name=aspect_beauty
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/aspect_beauty.out
#SBATCH --error=logs/aspect_beauty.err

python aspect_processing.py\
    --model_name sentence-transformers/sentence-t5-base \
    --dataset_path /data/common/RecommendationDatasets/Beauty_Amazon14/absa_filtered.csv \
    --output_dir /data/common/RecommendationDatasets/Beauty_Amazon14/ \
    --batch_size 32 \
    --threshold 0.6 \
    --min_community_size 5
