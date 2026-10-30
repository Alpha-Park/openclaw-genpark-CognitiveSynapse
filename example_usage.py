# -*- coding: utf-8 -*-
import time
from client import OpenClawGenParkCognitiveSynapse

synapse = OpenClawGenParkCognitiveSynapse()
now = time.time()

node_a = {
    "node_id": "fact_pricing_tier",
    "access_timestamps": [now - 300, now - 1800, now - 7200],
    "keywords": ["enterprise", "gpu", "pricing", "contract"]
}

# 1. ACT-R Activation
cues = ["enterprise", "pricing"]
act = synapse.compute_actr_activation(node_a, cues, current_timestamp=now)
print(f"ACT-R Activation: Score={act['total_activation_score']} (Retrieval Prob: {act['retrieval_probability']*100:.1f}%)")

# 2. Semantic Clustering
nodes = [
    {"node_id": "m1", "keywords": ["gpu", "cluster", "h100"]},
    {"node_id": "m2", "keywords": ["gpu", "cuda", "cluster"]},
    {"node_id": "m3", "keywords": ["invoice", "stripe", "billing"]},
    {"node_id": "m4", "keywords": ["billing", "accounting"]}
]
clusters = synapse.cluster_associative_semantic_graph(nodes, similarity_threshold=0.3)
print(f"\nSemantic Clusters ({clusters['clusters_count']}): {clusters['semantic_clusters']}")

# 3. Memory Pruning
old_nodes = [
    {"node_id": "fresh", "access_timestamps": [now - 60], "keywords": ["test"]},
    {"node_id": "stale", "access_timestamps": [now - 86400 * 30], "keywords": ["ancient"]}
]
pruned = synapse.prune_faded_memories(old_nodes, retention_threshold=-1.5, current_timestamp=now)
print(f"\nPruned: {pruned['pruned_count']} nodes purged, {pruned['surviving_count']} survived")
