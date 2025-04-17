# Ben Kabongo
# Statement Extraction and Aspect-Based Sentiment Analysis Post Processing

# April 2025


import argparse
import ast
import json
import numpy as np
import os
import pandas as pd

from collections import defaultdict
from sklearn.cluster import AgglomerativeClustering
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", type=str)
parser.add_argument("--output_dir", type=str)
parser.add_argument("--cluster_algo", type=str, default="Agglomerative") # "Agglomerative""
parser.add_argument("--distance_threshold", type=float, default=0.4)
parser.add_argument("--num_threads", type=int, default=4)
config = parser.parse_args()
config.cluster_algo = config.cluster_algo.lower()


def os_environ(num_threads=4):
    os.environ["OMP_NUM_THREADS"] = f"{num_threads}"
    os.environ["OPENBLAS_NUM_THREADS"] = f"{num_threads}"
    os.environ["MKL_NUM_THREADS"] = f"{num_threads}"
    os.environ["VECLIB_MAXIMUM_THREADS"] = f"{num_threads}"
    os.environ["NUMEXPR_NUM_THREADS"] = f"{num_threads}"


def count(data_df):
    if os.path.exists(os.path.join(config.output_dir, "aspects.json")):
        print(f"[INFO] Loading aspects from {os.path.join(config.output_dir, 'aspects.json')}")
        with open(os.path.join(config.output_dir, "aspects.json"), "r") as f:
            return json.load(f)
        print("[INFO] Loaded aspects")
    
    aspects = {}
    for sample in tqdm(data_df["json_results"], desc="Count", total=len(data_df)):
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


def embed(aspects):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    aspect_names = list(aspects.keys())
    
    print("[Start] Embedding aspects")
    if os.path.exists(os.path.join(config.output_dir, "embeddings.npy")):
        print(f"[INFO] Loading embeddings from {os.path.join(config.output_dir, 'embeddings.npy')}")
        aspect_embeddings = np.load(os.path.join(config.output_dir, "embeddings.npy"))
        print("[INFO] Loaded embeddings")
    else:
        aspect_embeddings = model.encode(aspect_names, normalize_embeddings=True)
        np.save(os.path.join(config.output_dir, "embeddings.npy"), aspect_embeddings)
    print("[End] Embedding aspects")
    
    print("[Start] Calculating similarities")
    if os.path.exists(os.path.join(config.output_dir, "similarities.npy")):
        print(f"[INFO] Loading similarities from {os.path.join(config.output_dir, 'similarities.npy')}")
        aspect_similarities = np.load(os.path.join(config.output_dir, "similarities.npy"))
        print("[INFO] Loaded similarities")
    else:
        aspect_similarities = model.similarity(aspect_embeddings, aspect_embeddings)
        print(f"[INFO] Saving similarities to {os.path.join(config.output_dir, 'similarities.npy')}")
        np.save(os.path.join(config.output_dir, "similarities.npy"), aspect_similarities)
        print(f"[INFO] Similarities saved to {os.path.join(config.output_dir, 'similarities.npy')}")
    print("[End] Calculating similarities")
    
    return aspect_embeddings, aspect_similarities

    
def agglomerative_clustering(aspect_similarities, distance_threshold=0.4):
    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric='precomputed',
        linkage='average',
        distance_threshold=distance_threshold
    )
    labels = clustering.fit_predict(-aspect_similarities + 1)
    return labels


#def hdbscan_clustering(aspect_embeddings, min_cluster_size=1, min_samples=1):
#    clusterer = HDBSCAN(
#        metric='euclidean',
#       min_cluster_size=min_cluster_size,
#        min_samples=min_samples,
#    )
#    labels = clusterer.fit_predict(aspect_embeddings)
#    return labels


def clustering(aspect_similarities, distance_threshold=0.4):
    if os.path.exists(os.path.join(config.output_dir, f"{config.cluster_algo}_labels.npy")):
        labels = np.load(os.path.join(config.output_dir, f"{config.cluster_algo}_labels.npy"))
        return labels
    
    if config.cluster_algo == "agglomerative":
        print("[Start] Agglomerative Clustering")
        labels = agglomerative_clustering(aspect_similarities, distance_threshold)
        print("[End] Agglomerative Clustering")
    else:
#        labels = hdbscan_clustering(aspect_embeddings, min_cluster_size, min_samples)
        raise ValueError(f"Unknown clustering algorithm: {config.cluster_algo}")
    print(f"[INFO] Saving labels to {os.path.join(config.output_dir, f'{config.cluster_algo}_labels.npy')}")
    np.save(os.path.join(config.output_dir, f"{config.cluster_algo}_labels.npy"), labels)
    print(f"[INFO] Labels saved to {os.path.join(config.output_dir, f'{config.cluster_algo}_labels.npy')}")
    return labels


def group_aspects(aspects, labels):
    if os.path.exists(os.path.join(config.output_dir, f"{config.cluster_algo}_clusters.json")):
        print(f"[INFO] Loading clusters from {os.path.join(config.output_dir, f'{config.cluster_algo}_clusters.json')}")
        with open(os.path.join(config.output_dir, f"{config.cluster_algo}_clusters.json"), "r") as f:
            return json.load(f)
        print("[INFO] Loaded clusters")
        
    aspect_names = list(aspects.keys())
    cluster_map = defaultdict(list)
    for idx, label in enumerate(labels):
        if label == -1:
            continue
        cluster_map[label].append(aspect_names[idx])

    grouped_aspects = {}
    for group_id, terms in tqdm(cluster_map.items(), desc="Group Aspects", total=len(cluster_map)):
        terms_sorted = sorted(terms, key=lambda t: -aspects[t])
        total_count = sum(aspects[t] for t in terms)
        representative = terms_sorted[0]
        grouped_aspects[representative] = {
            "aspects": terms_sorted,
            "total_count": total_count
        }

    grouped_aspects = dict(sorted(grouped_aspects.items(), key=lambda x: -x[1]["total_count"]))
    cluster_path = os.path.join(config.output_dir, f"{config.cluster_algo}_clusters.json")
    print(f"[INFO] Saving clusters to {cluster_path}")
    with open(cluster_path, "w") as f:
        json.dump(grouped_aspects, f, indent=4)
    print(f"[INFO] Clusters saved to {cluster_path}")

    return grouped_aspects


if __name__ == "__main__":
    os_environ(config.num_threads)
    os.makedirs(config.output_dir, exist_ok=True)
    data_df = pd.read_csv(config.dataset_path)
    aspects = count(data_df)
    _, aspect_similarities = embed(aspects)
    labels = clustering(aspect_similarities, config.distance_threshold)
    group_aspects(aspects, labels)
