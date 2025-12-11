"""
Full integrated RAG experiment limited to Natural Questions (development/validation).
- SentenceTransformer embeddings
- PCA
- Milvus: HNSW, IVF_PQ
- Optional hnswlib
- Cross-encoder reranking
- Prompt construction + optional Gemma (Vertex AI) generation
- Metrics: MRR, P@k, R@k, F1, p95 latency, QPS, indexing time, memory delta

Run: python run_nq_rag.py
Note: Requires a running Milvus server (standalone) on localhost:19530.
Install pymilvus: pip install pymilvus
"""

import os
import time
import json
import numpy as np
import psutil
from tqdm import tqdm
from typing import List, Tuple, Dict, Any
from sklearn.decomposition import PCA

# set envs to avoid TF if not needed
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")

# --------------------------- 
# Config
# ---------------------------
CONFIG = {
    "model_name": "all-mpnet-base-v2",
    "cross_encoder_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "dataset": "trivia_qa",
    "dataset_split": "train",       # use development set (validation)
    "max_docs": 138384,                   # reduce size for testing; adjust as needed
    # "max_docs": 1383,                   # reduce size for testing; adjust as needed
    "min_docs": 100,                   # reduce size for testing; adjust as needed
    "pca_dim": 64,
    "pq_m": 8,
    "pq_bits": 8,
    "hnsw_m": 32,
    "hnsw_ef_construction": 100,
    "hnsw_ef_search": 50,
    "ivf_nlist": 128,                   # for IVF_PQ
    "ivf_nprobe": 10,                   # for IVF_PQ search
    "k_retrieval": 5,
    "rerank_k": 10,
    "do_rerank": True,
    "user_query_instruction": "Answer concisely using only retrieved passages. Cite passages by their index in brackets.",
    # "llm_model_name": "google/flan-t5-large",
    "llm_model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    # "llm_model_name": "google/gemma-3-1b-it",
    "llm_max_tokens": 256,
    "random_seed": 42,
    "save_report": "tinyLlama-1_tqa_rag_report.json",
    "milvus_host": "localhost",
    "milvus_port": "19530"
}

np.random.seed(CONFIG["random_seed"])

# ---------------------------
# Safe imports (may install missing packages manually)
# ---------------------------
try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
except Exception as e:
    raise RuntimeError("Install sentence-transformers (`pip install sentence-transformers`) before running.") from e

try:
    from datasets import load_dataset
except Exception as e:
    raise RuntimeError("Install datasets (`pip install datasets`) before running.") from e

try:
    from pymilvus import (
        connections,
        utility,
        FieldSchema, CollectionSchema, DataType, Collection,
    )
except Exception as e:
    raise RuntimeError("Install pymilvus (`pip install pymilvus`) before running. Ensure Milvus server is running on localhost:19530.") from e

try:
    import hnswlib
    HAS_HNSWLIB = True
except Exception:
    HAS_HNSWLIB = False
    
    
import torch

def get_device():
    if torch.cuda.is_available():
        print("[INFO] Using GPU:", torch.cuda.get_device_name(0))
        return "cuda"
    else:
        print("[INFO] GPU not available. Using CPU.")
        return "cpu"

DEVICE = get_device()

# ---------------------------
# Utilities
# ---------------------------
def mem_usage_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024*1024)

def elapsed_ms(start, end):
    return (end - start) * 1000.0

def normalize_L2(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (norms + 1e-10)

def mrr_at_k(retrieved_ids, ground_truth_indices, k):
    rr_sum = 0.0
    n = len(ground_truth_indices)
    for i in range(n):
        gt = ground_truth_indices[i]
        if gt == -1:
            continue
        rlist = retrieved_ids[i][:k]
        if gt in rlist:
            rank = rlist.index(gt) + 1
            rr_sum += 1.0 / rank
    return rr_sum / max(1, n)

def precision_at_k(retrieved_ids, ground_truth_indices, k):
    total_prec = 0.0
    n = len(ground_truth_indices)
    for i in range(n):
        gt = ground_truth_indices[i]
        if gt == -1:
            continue
        rlist = retrieved_ids[i][:k]
        total_prec += 1.0 if gt in rlist else 0.0
    return total_prec / max(1, n)

def recall_at_k(retrieved_ids, ground_truth_indices, k):
    # in this simplified one-gt case recall==precision (since single ground truth per query)
    total_rec = 0.0
    n = len(ground_truth_indices)
    for i in range(n):
        gt = ground_truth_indices[i]
        if gt == -1:
            continue
        rlist = retrieved_ids[i][:k]
        total_rec += 1.0 if gt in rlist else 0.0
    return total_rec / max(1, n)

def f1_from_pr(p, r):
    return 2 * p * r / (p + r + 1e-8)

# ---------------------------
# Load Natural Questions (validation)
# Robust extraction of passage & query + create ground-truth mapping
# ---------------------------
def load_triviaqa_dev(max_docs: int):
    """
    Load TriviaQA dev/validation data using the top search result as the document text.
    """
    ds = load_dataset("trivia_qa", "rc", split=CONFIG["dataset_split"])
    docs, queries, ground_truth = [], [], []

    for idx, ex in enumerate(tqdm(ds, desc="Loading TriviaQA dev")):
        if len(docs) >= max_docs:
            break

        # 1) Question
        q = ex.get("question")
        if not q or len(q.strip()) == 0:
            continue

        # 2) Top search result as document
        doc_text = ""
        search_results = ex.get("search_results", {})
        if search_results and "search_context" in search_results:
            doc_text = search_results["search_context"]
            # If it's a list, join into one string
            if isinstance(doc_text, list):
                doc_text = " ".join(doc_text)

        # skip very short docs
        if not doc_text or len(doc_text.strip()) < 20:
            continue


        docs.append(doc_text)
        queries.append(q)
        ground_truth.append(len(docs) - 1)

    print(f"Loaded {len(docs)} docs & queries from TriviaQA ({CONFIG['dataset_split']}) using top search result.")
    return docs, queries, ground_truth

# ---------------------------
# Embedding helper
# ---------------------------
def embed_texts(model, texts: List[str], batch_size: int = 64) -> np.ndarray:
    if not texts:
        dim = model.get_sentence_embedding_dimension()
        return np.empty((0, dim), dtype='float32')
    # embs = model.encode(texts, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=True)
    # GPU-accelerated encoding
    embs = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=True,
        device=DEVICE,               # <<— force GPU
    )
    embs = np.atleast_2d(embs)
    embs = np.ascontiguousarray(embs.astype('float32'))
    # manual L2 normalization
    embs = normalize_L2(embs)
    return embs

# ---------------------------
# PCA
# ---------------------------
def fit_pca(x: np.ndarray, n_components: int):
    print(f"Fitting PCA {x.shape[1]} -> {n_components}")
    pca = PCA(n_components=n_components, random_state=CONFIG["random_seed"])
    x_reduced = pca.fit_transform(x)
    x_reduced = np.ascontiguousarray(x_reduced.astype('float32'))
    x_reduced = normalize_L2(x_reduced)
    return pca, x_reduced

# ---------------------------
# Milvus connection
# ---------------------------
def connect_milvus():
    connections.connect(
        alias="default",
        host=CONFIG["milvus_host"],
        port=CONFIG["milvus_port"]
    )
    print(f"Connected to Milvus at {CONFIG['milvus_host']}:{CONFIG['milvus_port']}")

# ---------------------------
# Milvus builders
# ---------------------------
def build_milvus_hnsw(vectors: np.ndarray, collection_name: str) -> Tuple[Collection, float, float, Dict[str, Any]]:
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)
    
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=vectors.shape[1])
    ]
    schema = CollectionSchema(fields, description="HNSW collection")
    collection = Collection(name=collection_name, schema=schema)
    
    ids = np.arange(len(vectors), dtype=np.int64)
    data = [ids.tolist(), vectors.tolist()]  # Convert to lists for insertion
    
    t0 = time.time()
    mem_before = mem_usage_mb()
    collection.insert(data)
    collection.flush()
    
    index_params = {
        "metric_type": "COSINE",
        "index_type": "HNSW",
        "params": {"M": CONFIG["hnsw_m"], "efConstruction": CONFIG["hnsw_ef_construction"]}
    }
    collection.create_index(field_name="embedding", index_params=index_params)
    collection.load()
    
    t1 = time.time()
    mem_after = mem_usage_mb()
    build_time = elapsed_ms(t0, t1)
    mem_delta = max(0.0, mem_after - mem_before)
    search_params = {"ef": CONFIG["hnsw_ef_search"]}
    
    print(f"Built Milvus HNSW collection '{collection_name}'.")
    return collection, build_time, mem_delta, search_params

def build_milvus_ivf_pq(vectors: np.ndarray, collection_name: str) -> Tuple[Collection, float, float, Dict[str, Any]]:
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)
    
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=vectors.shape[1])
    ]
    schema = CollectionSchema(fields, description="IVF_PQ collection")
    collection = Collection(name=collection_name, schema=schema)
    
    ids = np.arange(len(vectors), dtype=np.int64)
    data = [ids.tolist(), vectors.tolist()]  # Convert to lists for insertion
    
    t0 = time.time()
    mem_before = mem_usage_mb()
    collection.insert(data)
    collection.flush()
    
    index_params = {
        "metric_type": "COSINE",
        "index_type": "IVF_PQ",
        "params": {"nlist": CONFIG["ivf_nlist"], "m": CONFIG["pq_m"], "nbits": CONFIG["pq_bits"]}
    }
    collection.create_index(field_name="embedding", index_params=index_params)
    collection.load()
    
    t1 = time.time()
    mem_after = mem_usage_mb()
    build_time = elapsed_ms(t0, t1)
    mem_delta = max(0.0, mem_after - mem_before)
    search_params = {"nprobe": CONFIG["ivf_nprobe"]}
    
    print(f"Built Milvus IVF_PQ collection '{collection_name}'.")
    return collection, build_time, mem_delta, search_params

def build_hnswlib(vectors: np.ndarray):
    if not HAS_HNSWLIB:
        return None
    dim = vectors.shape[1]
    p = hnswlib.Index(space='cosine', dim=dim)
    p.init_index(max_elements=vectors.shape[0], ef_construction=CONFIG["hnsw_ef_construction"], M=CONFIG["hnsw_m"])
    p.set_ef(CONFIG["hnsw_ef_search"])
    p.add_items(vectors, np.arange(vectors.shape[0]))
    return p

# ---------------------------
# Retrieval & evaluation helpers
# ---------------------------
def retrieve_milvus(collection: Collection, q_emb: np.ndarray, k: int, search_params: Dict[str, Any]):
    start = time.time()
    res = collection.search(
        data=[q_emb.tolist()],  # Convert query to list
        anns_field="embedding",
        param=search_params,
        limit=k,
        output_fields=["id"]
    )
    ids = [hit.entity.get("id") for hit in res[0]]
    end = time.time()
    return ids, elapsed_ms(start, end)

def evaluate_milvus(collection: Collection, search_params: Dict[str, Any], q_embs: np.ndarray, ground_truth: List[int], k: int):
    latencies = []
    retrieved_ids = []
    for q in q_embs:
        ids, latency_ms = retrieve_milvus(collection, q, k, search_params)
        latencies.append(latency_ms)
        retrieved_ids.append(ids)
    p95_latency = np.percentile(latencies, 95) if len(latencies) > 0 else 0.0
    qps = 1000.0 / np.mean(latencies) if np.mean(latencies) > 0 else 0.0
    mrr = mrr_at_k(retrieved_ids, ground_truth, k)
    p = precision_at_k(retrieved_ids, ground_truth, k)
    r = recall_at_k(retrieved_ids, ground_truth, k)
    f1 = f1_from_pr(p, r)
    return {
        "p95_latency": float(p95_latency),
        "QPS": float(qps),
        "MRR": float(mrr),
        "P": float(p),
        "R": float(r),
        "F1": float(f1)
    }

def evaluate_hnswlib(idx, q_embs: np.ndarray, ground_truth: List[int], k: int):
    if idx is None:
        return None
    latencies = []
    retrieved_ids = []
    for q in q_embs:
        start = time.time()
        labels, distances = idx.knn_query(q.reshape(1, -1), k=k)
        end = time.time()
        latencies.append(elapsed_ms(start, end))
        retrieved_ids.append(labels[0].tolist())
    p95_latency = np.percentile(latencies, 95) if len(latencies) > 0 else 0.0
    qps = 1000.0 / np.mean(latencies) if np.mean(latencies)>0 else 0.0
    mrr = mrr_at_k(retrieved_ids, ground_truth, k)
    p = precision_at_k(retrieved_ids, ground_truth, k)
    r = recall_at_k(retrieved_ids, ground_truth, k)
    f1 = f1_from_pr(p, r)
    return {"p95_latency": float(p95_latency), "QPS": float(qps), "MRR": float(mrr), "P": float(p), "R": float(r), "F1": float(f1)}

# ---------------------------
# Cross-encoder rerank + prompt + LLM
# ---------------------------
def rerank_with_crossencoder(cross_encoder, query, candidate_texts, candidate_ids):
    """
    Rerank candidate documents using a CrossEncoder.

    Args:
        cross_encoder: a SentenceTransformer CrossEncoder model
        query: str, the query text
        candidate_texts: list of str, candidate document texts
        candidate_ids: list of int or str, corresponding candidate document IDs

    Returns:
        sorted_ids: list of candidate IDs sorted by score descending
        sorted_scores: list of corresponding scores
    """

    # Ensure query is a string
    if not isinstance(query, str):
        query = str(query)

    # Ensure candidate_texts is a list of strings
    candidate_texts = [str(c) for c in candidate_texts]

    # Build pairs for cross-encoder
    pairs = [(query, c) for c in candidate_texts]

    # Predict scores
    scores = cross_encoder.predict(pairs, batch_size=32)  # higher is better

    # Sort by score descending
    sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    sorted_ids = [candidate_ids[i] for i in sorted_indices]
    sorted_scores = [scores[i] for i in sorted_indices]

    return sorted_ids, sorted_scores


PROMPT_TEMPLATE = """User instruction:
{instruction}

Retrieved passages:
{retrieved}

Task:
Using only the retrieved passages above, produce a concise answer to the user's query.
Start your answer with the answer text.
"""

def build_prompt(user_query: str):
    """
    Construct a prompt using actual passage text, not IDs/hashes.
    """
    # lines = []
    # for did, doc_text, score in retrieved_pairs[:max_passages]:
    #     excerpt = doc_text.replace("\n", " ")[:500]  # use real text
    #     lines.append(f"[{did}] {excerpt}")
    # retrieved_text = "\n\n".join(lines)
    prompt = f"Query: {user_query}\n\n"
    # prompt = f"Query: {user_query}\n\nUser instruction:\n{instruction}\n\nRetrieved passages:\n{retrieved_text}\n\nTask:\nUsing only the retrieved passages above, produce a concise answer to the user's query. Start your answer with the answer text"
    return prompt
# def build_prompt(user_query: str, retrieved_pairs: List[Tuple[int, str, float]], instruction: str = CONFIG["user_query_instruction"], max_passages: int = 5):
#     """
#     Construct a prompt using actual passage text, not IDs/hashes.
#     """
#     lines = []
#     for did, doc_text, score in retrieved_pairs[:max_passages]:
#         excerpt = doc_text.replace("\n", " ")[:500]  # use real text
#         lines.append(f"[{did}] {excerpt}")
#     retrieved_text = "\n\n".join(lines)
#     prompt = f"Query: {user_query}\n\nUser instruction:\n{instruction}\n\nRetrieved passages:\n{retrieved_text}\n\nTask:\nUsing only the retrieved passages above, produce a concise answer to the user's query. Start your answer with the answer text"
#     return prompt

# ---------------------------
# Free LLM using Hugging Face
# ---------------------------
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def init_free_llm(model_name=CONFIG["llm_model_name"], device_map="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Load a free local/open-source LLM for text generation.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Use AutoModelForSeq2SeqLM for T5-like models
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device_map)
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return generator

def generate_with_free_llm(generator, prompt: str, max_tokens: int = 256):
    output = generator(prompt, max_new_tokens=max_tokens, do_sample=True, temperature=0.7)
    return output[0]["generated_text"]

# ---------------------------
# Evaluate index with rerank (per-index)
# ---------------------------
def evaluate_index_with_rerank(index_name: str, index_obj: Dict[str, Any], docs: List[str], queries_text: List[str], query_embs: np.ndarray, ground_truth: List[int], cross_encoder=None, k=5, rerank_k=10):
    collection = index_obj["collection"]
    search_params = index_obj["search_params"]
    nq = len(queries_text)
    raw_retrieved = []
    raw_latencies = []
    reranked_retrieved = []
    retrieval_plus_rerank_latencies = []

    # we will keep for optional prompt generation: top reranked pairs for first query
    per_query_reranked_pairs = []

    for qi in range(nq):
        q_emb = query_embs[qi]
        q_text = queries_text[qi]
        to_retrieve = max(k, rerank_k)
        ids, latency_ms = retrieve_milvus(collection, q_emb, to_retrieve, search_params)
        raw_retrieved.append(ids[:k])
        raw_latencies.append(latency_ms)

        if cross_encoder is not None and CONFIG["do_rerank"]:
            candidate_ids = ids[:to_retrieve]
            candidate_texts = [docs[cid] for cid in candidate_ids]
            t0 = time.time()
            sorted_ids, sorted_scores = rerank_with_crossencoder(cross_encoder, q_text, candidate_texts, candidate_ids)
            t1 = time.time()
            retrieval_plus_rerank_latencies.append(latency_ms + elapsed_ms(t0, t1))
            reranked_pairs = [(sid, docs[sid], sc) for sid, sc in zip(sorted_ids, sorted_scores)]
            reranked_retrieved.append([p[0] for p in reranked_pairs[:k]])
            per_query_reranked_pairs.append(reranked_pairs)
        else:
            retrieval_plus_rerank_latencies.append(latency_ms)
            reranked_retrieved.append(ids[:k])
            per_query_reranked_pairs.append([(i, docs[i], 0.0) for i in ids[:k]])

    def compute_metrics(retrieved_lists, latencies_ms):
        p95_latency = float(np.percentile(latencies_ms, 95)) if len(latencies_ms)>0 else 0.0
        qps = float(1000.0 / np.mean(latencies_ms)) if np.mean(latencies_ms) > 0 else 0.0
        mrr = mrr_at_k(retrieved_lists, ground_truth, k)
        p = precision_at_k(retrieved_lists, ground_truth, k)
        r = recall_at_k(retrieved_lists, ground_truth, k)
        f1 = f1_from_pr(p, r)
        return {"p95_latency": p95_latency, "QPS": qps, "MRR": mrr, "P": p, "R": r, "F1": f1}

    before_metrics = compute_metrics(raw_retrieved, raw_latencies)
    after_metrics = compute_metrics(reranked_retrieved, retrieval_plus_rerank_latencies)

    return {
        "index": index_name,
        "before_rerank": before_metrics,
        "after_rerank": after_metrics,
        "raw_latencies_ms": raw_latencies,
        "retrieval_plus_rerank_latencies_ms": retrieval_plus_rerank_latencies,
        "per_query_reranked_pairs": per_query_reranked_pairs
    }
    
    
# UTILS
def format_metrics(metrics_dict):
    """
    Add units to metrics for readability.
    """
    formatted = {}
    for k, v in metrics_dict.items():
        if k in ["p95_latency"]:
            formatted[k] = f"{v:.2f} ms"
        elif k in ["QPS"]:
            formatted[k] = f"{v:.2f} queries/s"
        elif k in ["MRR", "P", "R", "F1"]:
            formatted[k] = f"{v*100:.2f}"  # as percentage
        else:
            formatted[k] = v
    return formatted

# ---------------------------
# Run experiment main
# ---------------------------
def run_experiment():
    # 1) load data
    docs, queries, ground_truth = load_triviaqa_dev(CONFIG["max_docs"])
    if len(docs) == 0:
        raise RuntimeError("No docs loaded from TriviaQA dev. Check dataset loading.")
    # limit queries for faster test if too many
    # you can adjust or use all
    max_queries = min(CONFIG['min_docs'], len(queries))  # keep evaluation tractable
    queries_sample = queries[:max_queries]
    ground_truth_sample = ground_truth[:max_queries]

    # 2) connect to Milvus
    connect_milvus()

    # 3) load embedding model
    print("Loading embedding model:", CONFIG["model_name"])
    # embed_model = SentenceTransformer(CONFIG["model_name"])
    embed_model = SentenceTransformer(CONFIG["model_name"], device=DEVICE)


    # 4) embed docs and queries
    print("Embedding documents...")
    doc_embs = embed_texts(embed_model, docs)
    print("Embedding queries...")
    query_embs = embed_texts(embed_model, queries_sample)

    # 5) PCA reduce
    if CONFIG["pca_dim"] is not None and CONFIG["pca_dim"] < doc_embs.shape[1]:
        pca, doc_embs_reduced = fit_pca(doc_embs, CONFIG["pca_dim"])
        query_embs_reduced = normalize_L2(np.ascontiguousarray(pca.transform(query_embs).astype('float32')))
    else:
        doc_embs_reduced = doc_embs
        query_embs_reduced = query_embs
    
    # memory baseline
    total_mem = mem_usage_mb()
    mem_per_doc = total_mem / max(1, len(docs))
    print(f"Memory usage: {total_mem:.2f} MB, Memory/doc: {mem_per_doc:.6f} MB")

    # 6) Build indices
    index_stats = {}

    print("Building Milvus HNSW index...")
    milvus_hnsw_coll, hnsw_time, hnsw_mem, hnsw_search_params = build_milvus_hnsw(doc_embs_reduced, "nq_hnsw")
    index_stats["milvus_hnsw"] = {"index_time_ms": hnsw_time, "index_mem_mb": hnsw_mem}

    print("Building Milvus IVF_PQ index...")
    milvus_pq_coll, pq_time, pq_mem, pq_search_params = build_milvus_ivf_pq(doc_embs_reduced, "nq_ivf_pq")
    index_stats["milvus_ivf_pq"] = {"index_time_ms": pq_time, "index_mem_mb": pq_mem}

    hnswlib_idx = None
    if HAS_HNSWLIB:
        print("Building hnswlib index...")
        mem_before = mem_usage_mb(); t0 = time.time()
        hnswlib_idx = build_hnswlib(doc_embs_reduced)
        t1 = time.time(); mem_after = mem_usage_mb()
        index_stats["hnswlib"] = {"index_time_ms": elapsed_ms(t0, t1), "index_mem_mb": max(0.0, mem_after - mem_before)}

    print("Indexing summary:")
    for k, v in index_stats.items():
        print(f"  {k}: time={v['index_time_ms']:.2f}ms, Δmem={v['index_mem_mb']:.2f}MB")

    # 7) Optional cross-encoder
    cross_encoder = None
    if CONFIG["do_rerank"]:
        print("Loading cross-encoder:", CONFIG["cross_encoder_model"])
        cross_encoder = CrossEncoder(CONFIG["cross_encoder_model"])

    # 8) Evaluate indices with reranking
    print("Evaluating Milvus HNSW (with optional rerank)...")
    res_hnsw = evaluate_index_with_rerank(
        "milvus_hnsw", 
        {"collection": milvus_hnsw_coll, "search_params": hnsw_search_params}, 
        docs, queries_sample, query_embs_reduced, ground_truth_sample, 
        cross_encoder, k=CONFIG["k_retrieval"], rerank_k=CONFIG["rerank_k"]
    )
    res_hnsw["index_stats"] = index_stats["milvus_hnsw"]

    print("Evaluating Milvus IVF_PQ (with optional rerank)...")
    res_pq = evaluate_index_with_rerank(
        "milvus_ivf_pq", 
        {"collection": milvus_pq_coll, "search_params": pq_search_params}, 
        docs, queries_sample, query_embs_reduced, ground_truth_sample, 
        cross_encoder, k=CONFIG["k_retrieval"], rerank_k=CONFIG["rerank_k"]
    )
    res_pq["index_stats"] = index_stats["milvus_ivf_pq"]

    results = {
        "milvus_hnsw": res_hnsw,
        "milvus_ivf_pq": res_pq
    }

    if hnswlib_idx is not None and "hnswlib" in index_stats:
        print("Evaluating hnswlib...")
        res_hnswlib = evaluate_hnswlib(hnswlib_idx, query_embs_reduced, ground_truth_sample, CONFIG["k_retrieval"])
        results["hnswlib"] = {"index": "hnswlib", "before_rerank": res_hnswlib, "index_stats": index_stats.get("hnswlib", {})}

    # 9) Generate prompt + free LLM for a small sample query
    sample_qi = 0
    sample_query = queries_sample[sample_qi]
    sample_reranked_pairs = res_hnsw["per_query_reranked_pairs"][sample_qi] if res_hnsw and "per_query_reranked_pairs" in res_hnsw else []
    # prompt = build_prompt(sample_query, sample_reranked_pairs, instruction=CONFIG["user_query_instruction"], max_passages=5)
    prompt = build_prompt(sample_query)

    llm_output = None

    try:
        print("Initializing free LLM...")
        generator = init_free_llm(CONFIG["llm_model_name"])  # smaller/free model
        print("Generating answer with free LLM (sample)...")
        llm_output = generate_with_free_llm(generator, prompt, CONFIG["llm_max_tokens"])
        print("LLM output sample:\n", llm_output)
    except Exception as e:
        print("Free LLM call failed:", e)
        print("Prompt (printed instead):\n", prompt)


    # 10) Print a clean report + save JSON
    def generate_report_dict(results):
        simplified = {}
        for name, r in results.items():
            if name == "hnswlib":
                simplified[name] = {
                    "index_stats": {
                        "index_time_ms": f"{r.get('index_stats', {}).get('index_time_ms', 0):.2f} ms",
                        "index_mem_mb": f"{r.get('index_stats', {}).get('index_mem_mb', 0):.2f} MB"
                    },
                    "metrics": format_metrics(r.get("before_rerank", {}))
                }
                continue

            simplified[name] = {
                "index_stats": {
                    "index_time_ms": f"{r.get('index_stats', {}).get('index_time_ms', 0):.2f} ms",
                    "index_mem_mb": f"{r.get('index_stats', {}).get('index_mem_mb', 0):.2f} MB"
                },
                "before_rerank": format_metrics(r.get("before_rerank", {})),
                "after_rerank": format_metrics(r.get("after_rerank", {}))
            }
        return simplified


    report = generate_report_dict(results)
    # attach prompt + llm output
    report_meta = {
        "sample_query_index": sample_qi,
        "sample_query": sample_query,
        "sample_prompt": prompt,
        "llm_output": llm_output
    }
    final_report = {"config": CONFIG, "report": report, "meta": report_meta}

    with open(CONFIG["save_report"], "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2, ensure_ascii=False)
    print(f"Report saved to {CONFIG['save_report']}")

    # print summary
    print("\n===== SUMMARY =====")
    for idx_name, data in results.items():
        print(f"\nIndex: {idx_name}")
        if idx_name == "hnswlib":
            print("  Metrics (hnswlib):", data.get("before_rerank", {}))
            continue
        print("  Index time (ms):", data["index_stats"].get("index_time_ms", None))
        print("  Mem delta (MB)  :", data["index_stats"].get("index_mem_mb", None))
        print("  BEFORE rerank:", ", ".join([f"{k}={v}" for k, v in format_metrics(data['before_rerank']).items()]))
        print("  AFTER  rerank:", ", ".join([f"{k}={v}" for k, v in format_metrics(data['after_rerank']).items()]))


    return final_report

# ---------------------------
# Entrypoint
# ---------------------------
if __name__ == "__main__":
    report = run_experiment()