# -*- coding: utf-8 -*-
"""
OpenClaw GenPark Cognitive Synapse - ACT-R Spreading Activation Memory & Semantic Association Graph
Standard Library only: math, time, typing.
Calculates ACT-R base-level power-law decay retention and contextual spreading activation,
clusters memory concepts into connected semantic cliques, and executes automated memory consolidation.
"""

import math
import time
from typing import Dict, List, Any

class OpenClawGenParkCognitiveSynapse:
    def __init__(self, decay_param_d: float = 0.5, default_fan_penalty_s: float = 2.0):
        self.d = decay_param_d
        self.S = default_fan_penalty_s

    def compute_actr_activation(self, memory_node: Dict[str, Any],
                                context_cues: List[str],
                                current_timestamp: float = None) -> Dict[str, Any]:
        """
        Calculates ACT-R activation A_i = B_i + sum_j W_j * S_ji
        Base level: B_i = ln( sum_{k=1}^n (t - t_k)^{-d} )
        Associative strength: S_ji = S - ln(fan_j)
        """
        now = current_timestamp if current_timestamp is not None else time.time()
        access_times = memory_node.get("access_timestamps", [now - 3600.0])
        keywords = set(w.lower() for w in memory_node.get("keywords", []))

        # 1. Base-level learning
        sum_decay = 0.0
        for t_k in access_times:
            delta_sec = max(1.0, now - t_k)
            delta_min = delta_sec / 60.0 # scale to minutes
            sum_decay += (delta_min ** (-self.d))

        base_level_B = math.log(max(1e-6, sum_decay))

        # 2. Contextual spreading activation
        W_j = 1.0 / max(1, len(context_cues)) # equal source attention weighting
        spreading_sum = 0.0
        matched_cues = []

        for cue in context_cues:
            cue_clean = cue.lower().strip()
            if cue_clean in keywords:
                matched_cues.append(cue_clean)
                fan_j = max(2, len(keywords)) # fan approximation
                S_ji = self.S - math.log(fan_j)
                spreading_sum += W_j * max(0.0, S_ji)

        total_activation = base_level_B + spreading_sum

        # Retrieval probability via logistic: 1 / (1 + exp(-(A - tau) / s))
        tau = -1.0 # retrieval threshold
        s_param = 0.4
        retrieval_prob = 1.0 / (1.0 + math.exp(- (total_activation - tau) / s_param))

        return {
            "node_id": memory_node.get("node_id", "mem_anon"),
            "base_level_activation": round(base_level_B, 4),
            "spreading_activation": round(spreading_sum, 4),
            "total_activation_score": round(total_activation, 4),
            "matched_context_cues": matched_cues,
            "retrieval_probability": round(retrieval_prob, 4),
            "is_retrievable": total_activation >= tau
        }

    def cluster_associative_semantic_graph(self, nodes: List[Dict[str, Any]],
                                           similarity_threshold: float = 0.4) -> Dict[str, Any]:
        """
        Computes pairwise keyword Jaccard overlap and identifies connected semantic clusters.
        """
        n = len(nodes)
        adj = {nodes[i]["node_id"]: [] for i in range(n)}

        for i in range(n):
            for j in range(i + 1, n):
                k1 = set(w.lower() for w in nodes[i].get("keywords", []))
                k2 = set(w.lower() for w in nodes[j].get("keywords", []))
                union = k1.union(k2)
                sim = len(k1.intersection(k2)) / len(union) if union else 0.0
                if sim >= similarity_threshold:
                    id1 = nodes[i]["node_id"]
                    id2 = nodes[j]["node_id"]
                    adj[id1].append(id2)
                    adj[id2].append(id1)

        # Connected components via BFS
        visited = set()
        clusters = []

        for node in nodes:
            nid = node["node_id"]
            if nid not in visited:
                comp = []
                queue = [nid]
                visited.add(nid)
                while queue:
                    curr = queue.pop(0)
                    comp.append(curr)
                    for neighbor in adj.get(curr, []):
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)
                clusters.append(comp)

        return {
            "total_nodes_analyzed": n,
            "clusters_count": len(clusters),
            "similarity_threshold": similarity_threshold,
            "semantic_clusters": clusters
        }

    def prune_faded_memories(self, memory_nodes: List[Dict[str, Any]],
                             retention_threshold: float = -2.0,
                             current_timestamp: float = None) -> Dict[str, Any]:
        """
        Evaluates base-level retention across nodes and purges memories whose activation falls below threshold.
        """
        now = current_timestamp if current_timestamp is not None else time.time()
        surviving = []
        pruned = []

        for m in memory_nodes:
            eval_res = self.compute_actr_activation(m, context_cues=[], current_timestamp=now)
            act = eval_res["total_activation_score"]
            if act >= retention_threshold:
                surviving.append({"node_id": m.get("node_id"), "activation": act})
            else:
                pruned.append({"node_id": m.get("node_id"), "activation": act})

        return {
            "initial_memory_count": len(memory_nodes),
            "surviving_count": len(surviving),
            "pruned_count": len(pruned),
            "retention_threshold": retention_threshold,
            "surviving_nodes": surviving,
            "pruned_nodes": pruned
        }
