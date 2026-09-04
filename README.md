# Veritas — Enterprise Hybrid RAG System

Veritas is a production-grade Retrieval-Augmented Generation (RAG) system built over enterprise corporate documents (annual 10-K filings across multiple fiscal years, risk factor disclosures, and governance charters).

It combines **dense vector search** (TF-IDF or OpenAI embeddings stored in ChromaDB) and **sparse lexical search** (BM25Okapi) via **Reciprocal Rank Fusion (RRF)**. Answers are rigorously grounded in retrieved passages with `[source, page]` citations, citation deduplication, and deterministic refusal ("I don't know") when information is absent. The system features a modern Next.js interface styled to Apple Human Interface Guidelines, a FastAPI backend with strict CORS, and a two-tier evaluation harness.

---

## Architecture

```
                                    ┌────────────────────────┐
                                    │   Raw 10-K Filings     │
                                    │  (.txt, .pdf, .html)   │
                                    └───────────┬────────────┘
                                                │
                                                ▼
                                    ┌────────────────────────┐
                                    │ Sentence-Aware Chunking│
                                    │  (400 tok, 60 overlap) │
                                    └───────────┬────────────┘
                                                │
                       ┌────────────────────────┴────────────────────────┐
                       │                                                 │
                       ▼                                                 ▼
          ┌─────────────────────────┐                       ┌─────────────────────────┐
          │ Precomputed Embeddings  │                       │  BM25 Lexical Index     │
          │ (TF-IDF or OpenAI API)  │                       │  (Exact Tokens/Numbers) │
          └────────────┬────────────┘                       └────────────┬────────────┘
                       │                                                 │
                       ▼                                                 ▼
          ┌─────────────────────────┐                       ┌─────────────────────────┐
          │  ChromaDB Vector Store  │                       │   BM25Okapi Serialized  │
          └────────────┬────────────┘                       └────────────┬────────────┘
                       │                                                 │
                       └────────────────────────┬────────────────────────┘
                                                │
                                  Query ────────┼────────────────────────┐
                                                ▼                        ▼
                                    ┌───────────────────────┐┌───────────────────────┐
                                    │  Dense Vector Search  ││  BM25 Lexical Search  │
                                    └───────────┬───────────┘└───────────┬───────────┘
                                                │                        │
                                                └───────────┬────────────┘
                                                            │
                                                            ▼
                                                ┌────────────────────────┐
                                                │ Reciprocal Rank Fusion │
                                                │   RRF(d) = Σ 1/(60+r)  │
                                                └───────────┬────────────┘
                                                            │
                                        ┌───────────────────┴───────────────────┐
                                        ▼                                       ▼
                            ┌───────────────────────┐               ┌───────────────────────┐
                            │ Deduplicated Citations│               │  Grounded Generation  │
                            │ unique by (src, page) │               │  (Mock/OpenAI/Claude) │
                            └───────────────────────┘               └───────────┬───────────┘
                                                                                │
                                                                                ▼
                                                                    ┌───────────────────────┐
                                                                    │ Grounded Answer with  │
                                                                    │ [source, page] & chips│
                                                                    └───────────────────────┘
```

---

## Key Features

- **Zero-Key Out-of-the-Box**: Uses `TfidfEmbeddingBackend` and `mock` generation provider by default. Requires zero API keys or external network calls to run full indexing, tests, and web UI.
- **Hybrid Retrieval & RRF**: Dense semantic search catches conceptual intent; BM25 catches exact numerical metrics ($185.0M, 14.5%, 3nm) and statutory citations. Fused with Reciprocal Rank Fusion ($k=60$).
- **Citation Deduplication**: User-facing citations are deduplicated by `(source, page)` preserving the highest RRF rank, while all retrieved passages remain in generation context.
- **Anti-Hallucination Refusal**: System prompt and mock extractive engine explicitly return *"I don't know based on the provided documents"* when questions fall outside the corpus.
- **Apple HIG Web UI**: Built with Next.js 14 App Router, Tailwind CSS, SF Pro typography, frosted glass materials (`backdrop-filter: blur(20px)`), slide-over settings sheet, and interactive citation popovers.
- **Production FastAPI Backend**: Strict CORS policies with explicit origin allowlists and clean HTTP 503 service handling.
- **Two-Tier Evaluation**: Deterministic offline metrics (Precision@5, Recall, Cross-Year Retrieval Rate, Refusal Rate) + opt-in RAGAS LLM-judged evaluation.

---

## Quickstart (Zero API Keys)

### 1. Clone & Setup Virtual Environment
```bash
git clone https://github.com/your-username/veritas-rag.git
cd veritas-rag

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

### 2. Build Indexes
Extracts multi-year 10-Ks from `data/raw/`, computes sentence-bounded chunks, and builds ChromaDB & BM25 indexes:
```bash
python -m src.embed_index
```

### 3. Query via CLI
```bash
python -m src.rag_pipeline "How did Acme's revenue change from 2024 to 2025?"
```

### 4. Run Pytest Suite
```bash
pytest tests/ -v
```

### 5. Run Evaluation Benchmark
```bash
python -m src.evaluate
```

---

## Running the Web Interface

### Start FastAPI Backend
```bash
source .venv/bin/activate
uvicorn api.main:app --host 127.0.0.1 --port 8000
```

### Start Next.js Frontend
In a separate terminal:
```bash
cd frontend
npm install
npm run dev
```
Open [http://localhost:3000](http://localhost:3000) in your browser.

---

## Docker Deployment

Veritas includes complete containerization with no obsolete `version:` tags in Compose:

```bash
# Build and launch both backend and frontend services
docker compose up --build -d

# Verify container health
docker compose ps

# View backend indexing and uvicorn logs
docker compose logs -f backend

# Stop services cleanly
docker compose down
```

---

## Evaluation Benchmark & Real Results

Veritas includes a two-tier evaluation harness:
1. **Tier 1 (Offline Deterministic)**: Evaluates Context Precision@5 and Context Recall against ground-truth sources across 6 standard benchmark questions (including multi-year comparative questions and unanswerable controls).
2. **Tier 2 (RAGAS LLM-Judged)**: Opt-in with `--ragas` to score Faithfulness and Answer Relevancy via an LLM judge.

### Actual Output from `python -m src.evaluate`
```
===========================================================================
VERITAS OFFLINE RETRIEVAL & GENERATION EVALUATION
===========================================================================
ID                         | Type               | Precision@5 | Recall | Sources Retrieved
---------------------------------------------------------------------------
q1_cross_year_revenue      | cross_fiscal_year  | 0.8000      | 1.0000 | acme_10k_2024_financials.txt, acme_10k_2025_financials.txt, acme_10k_2025_risk_factors.txt
q2_cross_year_margins      | cross_fiscal_year  | 0.8000      | 1.0000 | acme_10k_2024_financials.txt, acme_10k_2025_financials.txt, acme_10k_2025_risk_factors.txt
q3_supply_chain_risk       | risk_factors       | 0.4000      | 1.0000 | acme_10k_2025_financials.txt, acme_10k_2025_risk_factors.txt
q4_cybersecurity_limitation | risk_factors       | 0.2000      | 1.0000 | acme_10k_2024_financials.txt, acme_10k_2025_financials.txt, acme_10k_2025_risk_factors.txt, acme_governance_charter.txt
q5_audit_committee_governance | governance         | 0.4000      | 1.0000 | acme_10k_2024_financials.txt, acme_10k_2025_financials.txt, acme_governance_charter.txt
q6_unanswerable_quantum    | unanswerable       | 1.0000      | 1.0000 | Refusal verified: True
---------------------------------------------------------------------------
SUMMARY METRICS:
  • Mean Context Precision@5:  0.5200
  • Mean Context Recall:        1.0000
  • Cross-Year Retrieval Rate:  100.0% (2/2)
  • Unanswerable Refusal Rate:  100.0% (1/1)
===========================================================================
```

---

## Why Each Design Decision

### 1. Hybrid Retrieval vs. Dense-Only
Dense semantic embeddings map text into continuous geometric space, excelling at capturing high-level intent. However, dense models struggle with exact token matches such as dollar amounts (`$185.0 million`), percentages (`14.5%`), defined statutory terms (`Rule 10A-3`), and proper nouns. BM25 sparse search assigns high weights to rare tokens and excels at pinpoint numerical retrieval. Merging dense and sparse results guarantees that quantitative enterprise inquiries retrieve the exact clauses containing the queried metrics.

### 2. Reciprocal Rank Fusion (RRF) vs. Weighted Blending
Linear score combination ($\alpha \cdot \text{Dense} + (1-\alpha) \cdot \text{BM25}$) is fragile because cosine similarity and BM25 scores have fundamentally different scales and unbounded distributions. Normalizing them requires min-max tuning per query. In contrast, RRF ($RRF(d) = \sum \frac{1}{k + r(d)}$ with $k=60$) is purely ordinal, requires zero hyperparameter tuning across queries, and is the industry standard for robust hybrid fusion.

### 3. Pluggable Embedding Backends
By decoupling the `EmbeddingBackend` abstract interface, Veritas supports `TfidfEmbeddingBackend` as the default offline backend alongside `OpenAIEmbeddingBackend`. This ensures anyone can clone and immediately run the entire system without setting up external accounts or API keys.

### 4. Token Estimation Without External Tokenizers
`src/chunking.py` uses plain whitespace splitting with a $1.3\times$ scaling factor (`len(text.split()) * 1.3`). External tokenizers like `tiktoken` or `nltk` attempt to download model vocabulary files at runtime (`cl100k_base.tiktoken`, `nltk_data/tokenizers/punkt`), causing crashes in air-gapped, containerized, or restricted networks. A pure regex and whitespace approach ensures zero runtime network dependencies.

### 5. Precomputed Chroma Embeddings
Rather than relying on ChromaDB's built-in embedding functions (which attempt to connect to Hugging Face or external CDNs to download ONNX embedding models), Veritas computes embeddings explicitly through its own backend and passes them directly to `collection.add(embeddings=...)`. This guarantees that vector storage never makes unauthorized network calls.

### 6. Citation Deduplication
When documents are chunked with overlap or when a single page contains multiple chunks, naive RAG pipelines output repetitive citation chips (e.g., displaying `acme_10k_2025_risk_factors.txt · p.1` multiple times). Veritas passes all relevant chunks to the LLM context for maximum recall, but deduplicates user-facing citations by `(source, page)` in score-descending order, retaining only the highest-scoring citation per unique page.

**Example from real query:**
- Query: *"What are Acme's supply chain disruptions, component lead times, and semiconductor risks in 2025?"*
- Retrieved context passages: **5 chunks** (including both `acme_10k_2025_risk_factors.txt_p1_c0` and `acme_10k_2025_risk_factors.txt_p1_c1`)
- Deduplicated user citations: **4 citations** (`(acme_10k_2025_risk_factors.txt, page 1)` retained only once with highest score `0.032787`)

---

## Pulling Live SEC 10-K Filings

To ingest live filings from the SEC EDGAR system:
```bash
python -m scripts.fetch_sec_filings --ticker AAPL --count 2 --out data/raw
python -m src.embed_index
```
*(Note: Ensure `SEC_USER_AGENT` in `scripts/fetch_sec_filings.py` is configured with your organization name and email per SEC access policy).*

---

## Real API Key Configuration (Optional)

To enable OpenAI or Anthropic Claude generation:
1. Open `.env`
2. Set `LLM_PROVIDER=openai` or `LLM_PROVIDER=anthropic`
3. Add your API key:
   ```env
   OPENAI_API_KEY=sk-...
   # or
   ANTHROPIC_API_KEY=sk-ant-...
   ```
4. Consult provider documentation for active model identifiers:
   - [OpenAI Model Documentation](https://platform.openai.com/docs/models)
   - [Anthropic Model Documentation](https://docs.anthropic.com/en/docs/about-claude/models)
