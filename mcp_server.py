# -*- coding: utf-8 -*-
import sys, json
from client import OpenClawGenParkCognitiveSynapse

client = OpenClawGenParkCognitiveSynapse()

TOOLS = [
    {
        "name": "compute_actr_activation",
        "description": "Calculates ACT-R base-level learning and spreading activation score for a memory node.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "memory_node": {"type": "object"},
                "context_cues": {"type": "array", "items": {"type": "string"}},
                "current_timestamp": {"type": "number"}
            },
            "required": ["memory_node", "context_cues"]
        }
    },
    {
        "name": "cluster_associative_semantic_graph",
        "description": "Clusters memory nodes into connected semantic components based on keyword overlap.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "nodes": {"type": "array", "items": {"type": "object"}},
                "similarity_threshold": {"type": "number"}
            },
            "required": ["nodes"]
        }
    },
    {
        "name": "prune_faded_memories",
        "description": "Evaluates base-level retention and identifies faded memory nodes for garbage collection.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "memory_nodes": {"type": "array", "items": {"type": "object"}},
                "retention_threshold": {"type": "number"},
                "current_timestamp": {"type": "number"}
            },
            "required": ["memory_nodes"]
        }
    }
]

def handle_request(req):
    method = req.get("method")
    rid = req.get("id")
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": rid, "result": {"tools": TOOLS}}
    elif method == "tools/call":
        params = req.get("params", {})
        tname = params.get("name")
        args = params.get("arguments", {})
        if tname == "compute_actr_activation":
            res = client.compute_actr_activation(args["memory_node"], args["context_cues"], args.get("current_timestamp"))
            return {"jsonrpc": "2.0", "id": rid, "result": {"content": [{"type": "text", "text": json.dumps(res, indent=2)}]}}
        elif tname == "cluster_associative_semantic_graph":
            res = client.cluster_associative_semantic_graph(args["nodes"], args.get("similarity_threshold", 0.4))
            return {"jsonrpc": "2.0", "id": rid, "result": {"content": [{"type": "text", "text": json.dumps(res, indent=2)}]}}
        elif tname == "prune_faded_memories":
            res = client.prune_faded_memories(args["memory_nodes"], args.get("retention_threshold", -2.0), args.get("current_timestamp"))
            return {"jsonrpc": "2.0", "id": rid, "result": {"content": [{"type": "text", "text": json.dumps(res, indent=2)}]}}
        return {"jsonrpc": "2.0", "id": rid, "error": {"code": -32601, "message": "Method not found"}}
    return {"jsonrpc": "2.0", "id": rid, "error": {"code": -32600, "message": "Invalid Request"}}

def main():
    for line in sys.stdin:
        if not line.strip(): continue
        try:
            req = json.loads(line)
            resp = handle_request(req)
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()
        except Exception as e:
            sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": str(e)}}) + "\n")
            sys.stdout.flush()

if __name__ == "__main__":
    main()
