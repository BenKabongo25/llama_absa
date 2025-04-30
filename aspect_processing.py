# Ben Kabongo
# Statement Extraction and Aspect-Based Sentiment Analysis Post Processing

# April 2025


import argparse
import ast
import json
import logging
import numpy as np
import os
import pandas as pd
import torch

from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", type=str, default="")
parser.add_argument("--output_dir", type=str, default="")
parser.add_argument("--model_name", type=str, default="sentence-transformers/sentence-t5-base")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--threshold", type=float, default=0.6)
parser.add_argument("--min_community_size", type=int, default=5)
config = parser.parse_args()


def count(data_df, config):
    count_path = os.path.join(config.output_dir, "aspects.json")
    if os.path.exists(count_path):
        logging.info(f"Loading aspects from {count_path}")
        with open(count_path, "r") as f:
            return json.load(f)
        logging.info("Loaded aspects")
    
    aspects = {}
    for sample in tqdm(data_df["absa"], desc="Count", total=len(data_df)):
        try:
            entry = ast.literal_eval(sample)
        except:
            continue
        else:
            for entry_item in entry:
                aspect = entry_item.get("aspect")
                if aspect:
                    aspect = aspect.lower().strip()
                    aspects[aspect] = aspects.get(aspect, 0) + 1    
        
    sorted_aspects = dict(sorted(aspects.items(), key=lambda item: item[1], reverse=True))
    logging.info(f"Saving aspects to {count_path}")
    with open(count_path, "w") as f:
        json.dump(sorted_aspects, f, indent=4)
    logging.info(f"Aspects saved to {count_path}")
    return sorted_aspects


def embed(aspects, config):
    embedding_path = os.path.join(config.output_dir, "embeddings.npy")
    logging.info("[Start] Embedding aspects")

    if os.path.exists(embedding_path):
        logging.info(f"Loading embeddings from {embedding_path}")
        aspect_embeddings = np.load(embedding_path)
        logging.info("Loaded embeddings")
    
    else:
        aspect_names = list(aspects.keys())
        model = SentenceTransformer(config.model_name)
        model.eval()
        with torch.no_grad():
            aspect_embeddings = model.encode(
                aspect_names, normalize_embeddings=True, batch_size=config.batch_size, show_progress_bar=True
            )
        np.save(embedding_path, aspect_embeddings)
        logging.info(f"Embeddings saved to {embedding_path}")
    logging.info("[End] Embedding aspects")

    return aspect_embeddings


def clustering(aspects, aspect_embeddings, config):
    cluster_id_path = os.path.join(
        config.output_dir, f"aspect_clusters_ids_{config.threshold}_{config.min_community_size}.json"
    )
    if os.path.exists(cluster_id_path):
        logging.info(f"Loading clusters ids from {cluster_id_path}")
        with open(cluster_id_path, "r") as f:
            clusters_ids = json.load(f)
        logging.info("Loaded clusters ids")
    
    else:
        logging.info("[Start] Agglomerative Clustering")
        aspect_embeddings = aspect_embeddings.astype("float16")
        clusters_ids = util.community_detection(
            embeddings=aspect_embeddings,
            threshold=config.threshold,
            min_community_size=config.min_community_size,
            batch_size=config.batch_size,
            show_progress_bar=True,
        )
        logging.info("[End] Agglomerative Clustering")
    
        logging.info(f"Saving clusters ids to {cluster_id_path}")
        with open(cluster_id_path, "w") as f:
            json.dump(clusters_ids, f, indent=4)
        logging.info(f"Labels clusters ids to {cluster_id_path}")

    cluster_path = os.path.join(
        config.output_dir, f"aspect_clusters_{config.threshold}_{config.min_community_size}.json"
    )
    if os.path.exists(cluster_path):
        logging.info(f"Loading clusters from {cluster_path}")
        with open(cluster_path, "r") as f:
            clusters = json.load(f)
        logging.info("Loaded clusters")

    else:
        aspect_names = list(aspects.keys())
        clusters = {}
        for cluster_id, cluster in enumerate(clusters_ids):
            cluster_aspects = [aspect_names[i] for i in cluster]
            cluster_aspects = sorted(cluster_aspects, key=lambda t: -aspects[t])
            total_count = sum(aspects[t] for t in cluster_aspects)
            cluster_name = cluster_aspects[0]
            clusters[cluster_name] = {
                "total_count": total_count,
                "aspects": cluster_aspects
            }

        clusters_tmp = dict(sorted(clusters.items(), key=lambda x: -x[1]["total_count"]))
        clusters = {"clusters": clusters_tmp, "total_count": len(clusters_tmp)}
        logging.info(f"Saving clusters to {cluster_path}")
        with open(cluster_path, "w") as f:
            json.dump(clusters, f, indent=4)
        logging.info(f"Clusters saved to {cluster_path}")

    logging.info(f"Number of clusters: {len(clusters)}")
    logging.info("Clustering completed")
        
    return clusters_ids, clusters


if __name__ == "__main__":
    os.makedirs(config.output_dir, exist_ok=True)
    data_df = None
    if config.dataset_path != "":
        data_df = pd.read_csv(config.dataset_path)
    aspects = count(data_df, config)
    aspect_embeddings = embed(aspects, config)
    clusters_ids, clusters = clustering(aspects, aspect_embeddings, config)
