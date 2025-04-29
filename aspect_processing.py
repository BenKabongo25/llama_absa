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

from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", type=str, default="")
parser.add_argument("--output_dir", type=str, default="")
parser.add_argument("--model_name", type=str, default="sentence-transformers/sentence-t5-base")
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--threshold", type=float, default=0.6)
parser.add_argument("--min_community_size", type=int, default=5)
config = parser.parse_args()


def count(data_df, config):
    if os.path.exists(os.path.join(config.output_dir, "aspects.json")):
        print(f"[INFO] Loading aspects from {os.path.join(config.output_dir, 'aspects.json')}")
        with open(os.path.join(config.output_dir, "aspects.json"), "r") as f:
            return json.load(f)
        print("[INFO] Loaded aspects")
    
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
    count_path = os.path.join(config.output_dir, "aspects.json")
    print(f"[INFO] Saving aspects to {count_path}")
    with open(count_path, "w") as f:
        json.dump(sorted_aspects, f, indent=4)
    print(f"[INFO] Aspects saved to {count_path}")
    return sorted_aspects


def embed(aspects, config):
    model = SentenceTransformer(config.model_name)
    aspect_names = list(aspects.keys())
    
    print("[Start] Embedding aspects")
    if os.path.exists(os.path.join(config.output_dir, "embeddings.npy")):
        print(f"[INFO] Loading embeddings from {os.path.join(config.output_dir, 'embeddings.npy')}")
        aspect_embeddings = np.load(os.path.join(config.output_dir, "embeddings.npy"))
        print("[INFO] Loaded embeddings")
    else:
        aspect_embeddings = model.encode(aspect_names, normalize_embeddings=True, batch_size=config.batch_size, show_progress_bar=True)
        np.save(os.path.join(config.output_dir, "embeddings.npy"), aspect_embeddings)
    print("[End] Embedding aspects")
    print(f"[INFO] Embeddings saved to {os.path.join(config.output_dir, 'embeddings.npy')}")

    return aspect_embeddings


def clustering(aspects, aspect_embeddings, config):
    cluster_id_path = os.path.join(config.output_dir, f"aspect_clusters_ids_{config.threshold}_{config.min_community_size}.json")
    if os.path.exists(cluster_id_path):
        print(f"[INFO] Loading clusters ids from {cluster_id_path}")
        with open(cluster_id_path, "r") as f:
            clusters_ids = json.load(f)
        print("[INFO] Loaded  clusters ids")
    
    else:
        print("[Start] Agglomerative Clustering")
        clusters_ids = util.community_detection(
            embeddings=aspect_embeddings,
            threshold=config.threshold,
            min_community_size=config.min_community_size,
            batch_size=config.batch_size,
            show_progress_bar=True,
        )
        print("[End] Agglomerative Clustering")
    
        print(f"[INFO] Saving  clusters ids to {cluster_id_path}")
        with open(cluster_id_path, "w") as f:
            json.dump(clusters_ids, f, indent=4)
        print(f"[INFO] Labels  clusters ids to {cluster_id_path}")

    cluster_path = os.path.join(config.output_dir, f"aspect_clusters_{config.threshold}_{config.min_community_size}.json")
    if os.path.exists(cluster_path):
        print(f"[INFO] Loading clusters from {cluster_path}")
        with open(cluster_path, "r") as f:
            clusters = json.load(f)
        print("[INFO] Loaded clusters")

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
        print(f"[INFO] Saving clusters to {cluster_path}")
        with open(cluster_path, "w") as f:
            json.dump(clusters, f, indent=4)
        print(f"[INFO] Clusters saved to {cluster_path}")
        
    return clusters_ids, clusters


if __name__ == "__main__":
    os.makedirs(config.output_dir, exist_ok=True)
    data_df = None
    if config.dataset_path != "":
        data_df = pd.read_csv(config.dataset_path)
    aspects = count(data_df, config)
    aspect_embeddings = embed(aspects, config)
    clusters_ids, clusters = clustering(aspects, aspect_embeddings, config)
    print("[INFO] Clustering completed")
