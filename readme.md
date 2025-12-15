# Integrated RAG Experiment on TriviaQA: Milvus Indexing + Cross-Reranking

End-to-end Retrieval-Augmented Generation (RAG) experiment comparing Milvus index types (HNSW, IVF_PQ), optional PCA dimensionality reduction, and Cross-Encoder reranking. Includes prompt construction and optional answer generation with a local Hugging Face LLM. Uses a subset of the TriviaQA Reading Comprehension dataset (dev/validation).

## ✨ Features

- Data: TriviaQA RC (subset of dev/validation)
- Embeddings: SentenceTransformers (all-mpnet-base-v2)
- Dimensionality Reduction: Optional PCA (e.g., 768 → 64) for speed/memory
- Vector DB (Milvus):
    - HNSW (high-performance retrieval)
    - IVF_PQ (high compression with product quantization)
- Optional baseline: In-memory hnswlib index
- Reranking: Cross-Encoder (cross-encoder/ms-marco-MiniLM-L-6-v2)
- LLM (optional): Local Hugging Face model (e.g., TinyLlama/TinyLlama-1.1B-Chat-v1.0)
- Metrics: MRR@k, P@k, R@k, F1, p95 latency, QPS, index time, memory delta
- Reports: Saves detailed JSON with before/after reranking metrics

## ⚙️ Prerequisites

- Python 3.8+
- Milvus standalone running at localhost:19530 (see Milvus docs)

## 📦 Installation

```bash
git clone <your-repo-link>
cd <your-repo-name>

# All-in
pip install sentence-transformers datasets scikit-learn pymilvus tqdm numpy psutil hnswlib transformers torch
```

## 🚀 How to Run

1. Start Milvus and ensure it’s reachable at localhost:19530.
2. Run:
     ```bash
     python run_nq_rag.py
     ```

## ⚙️ Configuration

Edit CONFIG in run_nq_rag.py:

| Setting               | Default                                  | Description                                          |
|-----------------------|-------------------------------------------|------------------------------------------------------|
| model_name            | all-mpnet-base-v2                         | SentenceTransformer for embeddings                   |
| cross_encoder_model   | cross-encoder/ms-marco-MiniLM-L-6-v2      | Cross-Encoder for reranking                          |
| max_docs              | 138384                                    | Maximum documents to index                           |
| n_eval_queries        | 100                                       | Number of evaluation queries (alias: min_docs)       |
| pca_dim               | 64                                        | Target dim after PCA (None to disable)               |
| k_retrieval           | 5                                         | Final documents returned                             |
| rerank_k              | 10                                        | Candidates retrieved for reranking                   |
| hnsw_m                | 32                                        | HNSW connectivity parameter                          |
| ivf_nlist             | 128                                       | IVF_PQ quantizer clusters                            |
| llm_model_name        | TinyLlama/TinyLlama-1.1B-Chat-v1.0        | LLM for optional generation                          |
| save_report           | tinyLlama-1_tqa_rag_report.json           | Output report path                                   |

## 📊 Interpreting Results

- MRR, Precision/Recall/F1: Retrieval quality (higher is better)
- p95 latency (ms): 95th percentile latency (lower is better)
- QPS: Queries per second (higher is better)
- Index time (ms): Time to build the index
- Memory delta (MB): Memory change during index build
- Metrics are reported before reranking (raw Milvus) and after reranking (Cross-Encoder).

Output is saved to CONFIG["save_report"] (default tinyLlama-1_tqa_rag_report.json).

### Sample Report Skeleton

```json
{
    "config": { "...": "..." },
    "report": {
        "milvus_hnsw": {
            "index_stats": {
                "index_time_ms": "X.XX",
                "index_mem_mb": "Y.YY"
            },
            "before_rerank": {
                "p95_latency_ms": "A.AA",
                "qps": "B.BB",
                "mrr": "C.CC",
                "precision_at_k": "...",
                "recall_at_k": "...",
                "f1": "..."
            },
            "after_rerank": {
                "p95_latency_ms": "D.DD",
                "qps": "E.EE",
                "mrr": "F.FF",
                "precision_at_k": "...",
                "recall_at_k": "...",
                "f1": "..."
            }
        },
        "milvus_ivf_pq": { "...": "..." }
    },
    "meta": {
        "sample_query": "...",
        "sample_prompt": "...",
        "llm_output": "..."
    }
}
```

## 🔍 Key Observations (typical)

- PCA lowers indexing time and memory with minimal impact on MRR/F1.
- Cross-Encoder reranking significantly improves MRR and F1.
- HNSW generally yields lower latency; IVF_PQ provides better compression.
- Full-pipeline performance is sensitive to rerank_k and k_retrieval.
