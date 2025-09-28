# NAIVE RAG Pipeline with OpenAI + Chroma

This repository implements a **Retrieval-Augmented Generation (RAG)** pipeline using:
- **OpenAI API** for embeddings and LLM completions
- **Chroma** (Cloud + Local PersistentClient) as the vector database
- **Explicit precomputation of embeddings** (no implicit embedding_function)

The system supports:
1. Document ingestion (from `.txt` files)
2. Text chunking with overlap
3. Embedding precomputation
4. Storage in Chroma (cloud + local disk persistence)
5. Semantic retrieval via k-NN
6. Answer synthesis using ranked evidence

---

## 📂 Pipeline Overview

### 1. Document Loading
```python
documents = load_documents_from_directory("news_articles")
```
Loads `.txt` files into memory as `{id, text}` dictionaries.

### 2. Chunking
```python
chunks = split_text(doc["text"], chunk_size=1000, chunk_overlap=30)
```
Splits long documents into overlapping segments to preserve context.

### 3. Embedding Precomputation
```python
embedding = get_openai_embedding("hello world")
```
All chunks are explicitly embedded with `text-embedding-3-small`, stored as NumPy vectors.

### 4. Storage in Chroma
- **Cloud collection**: persistent, multi-tenant
- **Local persistence**: stored in `./chroma_db`

```python
collection.add(
    ids=[doc["id"] for doc in chunked_documents],
    documents=[doc["text"] for doc in chunked_documents],
    embeddings=[doc["embedding"] for doc in chunked_documents]
)
```

### 5. Semantic Retrieval
```python
matches = query_documents("What threat do Google and OpenAI face?", n_results=3)
```
Performs similarity search (k-NN) over embeddings. Returns ranked chunks with distances.

### 6. Rank-Aware Answer Synthesis
```python
final_answer = answer_with_retrieval(
    question="What threat do Google and OpenAI face according to the memo?",
    n_results=3
)
```

This function:
- Embeds the query
- Retrieves top-k chunks
- Formats them with explicit ranking (Rank 1 emphasized)
- Passes ranked context to the LLM (gpt-4.1-nano)
- Produces a concise, grounded answer

---

## 🚀 Example

### Question
```
What threat do Google and OpenAI face according to the internal memo?
```

### Retrieved Chunks (preview)
```
[1] id=doc1_chunk1  distance=0.499
Text: OpenAI may be synonymous with machine learning now and Google ...
--------------------------------------------------------------------------------
[2] id=doc1_chunk2  distance=0.742
Text: The memo points out that in March, a leaked foundation model ...
--------------------------------------------------------------------------------
```

### Final Answer
```
According to the internal memo, Google and OpenAI face the threat of
rapidly advancing open-source LLMs. These projects evolve faster through
community collaboration, eroding the competitive moat once thought to
be guaranteed by proprietary scale and infrastructure.
```

---

## 🏗️ Architecture

**Preprocessing**
- Document ingestion
- Chunking with overlap

**Vectorization**
- Explicit embedding computation via OpenAI API

**Storage**
- Dual persistence: Chroma Cloud + local disk

**Retrieval**
- Semantic search with `query_embeddings`

**Synthesis**
- Rank-aware context construction
- Concise answer generation by LLM

---

## ⚙️ Requirements

**Python 3.9+**

**Dependencies:**
```bash
pip install openai chromadb numpy
```

**Environment variables:**
```bash
export OPENAI_API=your_openai_api_key
export CHROMADB_TOKEN=your_chroma_token
export CHROMA_TENANT=your_chroma_tenant
```

---
## 🖼️ RAG Pipeline Diagram

```text
         ┌──────────────────┐
         │   Raw Documents  │
         │  (.txt articles) │
         └─────────┬────────┘
                   │
                   ▼
         ┌──────────────────┐
         │   Chunking       │
         │ (fixed size +    │
         │   overlap)       │
         └─────────┬────────┘
                   │
                   ▼
         ┌──────────────────┐
         │  Embedding        │
         │  (OpenAI          │
         │  text-embedding)  │
         └─────────┬────────┘
                   │
          ┌────────┴─────────┐
          ▼                  ▼
 ┌──────────────────┐  ┌──────────────────┐
 │  Chroma Cloud     │  │  Chroma Local    │
 │  (multi-tenant DB)│  │  ./chroma_db     │
 └─────────┬────────┘  └─────────┬────────┘
           │                      │
           ▼                      ▼
       ┌────────────────────────────────┐
       │  Semantic Retrieval (k-NN)     │
       │  query_embeddings vs database  │
       └─────────────────┬──────────────┘
                         │
                         ▼
       ┌────────────────────────────────┐
       │   Rank-Aware Context Builder   │
       │   (Rank 1 prioritized)         │
       └─────────────────┬──────────────┘
                         │
                         ▼
       ┌────────────────────────────────┐
       │   LLM Answer Synthesis         │
       │   (OpenAI GPT-4.1-nano)        │
       └────────────────────────────────┘
