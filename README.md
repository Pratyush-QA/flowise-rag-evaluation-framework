# Flowise RAG Evaluation Framework

A production-grade **RAGAS-based evaluation framework** for testing Retrieval-Augmented Generation (RAG) pipelines built with [Flowise](https://flowiseai.com/). Designed to validate response quality, context precision, faithfulness, and factual correctness of AI-generated answers — the way a Senior SDET would.

---

## What This Framework Tests

| Metric | What It Measures | Threshold |
|---|---|---|
| **Faithfulness** | Is the response grounded in retrieved context? (detects hallucinations) | >= 0.8 |
| **Response Relevancy** | Is the answer on-topic and complete? | >= 0.7 |
| **Factual Correctness** | Do the facts in the response match ground truth? | >= 0.7 |
| **Context Precision** | Are retrieved chunks actually relevant to the query? | >= 0.7 |
| **Context Recall** | Does retrieved context cover all info from ground truth? | >= 0.7 |
| **Rubric Score** | Custom criteria-based 1-5 scoring for response quality | >= 4/5 |
| **Topic Adherence** | Does a multi-turn conversation stay on topic? | >= 0.7 |
| **Multiple Metrics** | Runs 5 metrics at once; saves CSV + JSON report locally | All thresholds |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                        RAG Pipeline (Flowise)                        │
│                                                                      │
│  PDF Loader → Text Splitter → OpenAI Embeddings → ChromaDB           │
│                                    ↓                                 │
│                           ChatOpenAI (GPT-4o)                        │
│                                    ↓                                 │
│                    Conversational Retrieval Chain                    │
└──────────────────────────────────────────────────────────────────────┘
                                     │
                              HTTP /prediction API
                                     │
┌──────────────────────────────────────────────────────────────────────┐
│                     RAGAS Evaluation Framework                       │
│                                                                      │
│  flowise_client.py  →  conftest.py  →  Individual Metric Tests       │
│                                              ↓                       │
│                                   pytest + pytest-asyncio            │
│                                              ↓                       │
│                                    reports/ (CSV + JSON)             │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
Flowise_RAG_Evaluation/
├── conftest.py              # Shared pytest fixtures (LLM wrapper, chatflow ID)
├── flowise_client.py        # Flowise REST API client
├── pytest.ini               # pytest config: asyncio_mode=auto, custom markers
│
├── ContextPrecision.py      # Test: context_precision metric
├── ContextRecall.py         # Test: context_recall metric
├── Faithfulness.py          # Test: faithfulness (hallucination detection)
├── FactualCorrectness.py    # Test: factual_correctness vs ground truth
├── ResponseRelevancy.py     # Test: response_relevancy metric
├── RubricScore.py           # Test: custom rubric 1-5 score
├── TopicAdherence.py        # Test: multi-turn topic adherence
├── MultipleMetrics.py       # Test: 5 metrics in one run, saves report
│
├── TestGen.py               # RAGAS synthetic test set generator
├── create_test_doc.py       # Utility: generates Sample_Docs/scientists_knowledge_base.pdf
│
├── Sample_Docs/
│   ├── scientists_knowledge_base.pdf  # Knowledge base uploaded to Flowise
│   └── scientists.md                  # Markdown version for TestGen
│
├── reports/                 # Auto-generated test results (gitignored)
│   ├── multi_metrics_<timestamp>.csv
│   └── multi_metrics_<timestamp>.json
│
├── .env.example             # Template for environment variables
├── requirements.txt         # Python dependencies
└── .gitignore
```

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | >= 3.9 | |
| Node.js | >= 18 | For running Flowise |
| Flowise | Latest | `npm install -g flowise` |
| ChromaDB | Latest | `pip install chromadb` |
| OpenAI API Key | — | GPT-4o for LLM + embeddings |

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/Pratyush-QA/flowise-rag-evaluation-framework.git
cd flowise-rag-evaluation-framework
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Set up environment variables

```bash
cp .env.example .env
```

Edit `.env`:

```env
OPENAI_API_KEY=sk-...
FLOWISE_API_URL=http://localhost:3000
FLOWISE_CHATFLOW_ID=your-chatflow-id-here
FLOWISE_API_KEY=                          # Leave blank if no auth
```

### 4. Start ChromaDB (vector database)

```bash
chroma run --host localhost --port 8000
```

### 5. Start Flowise

```bash
flowise start
# Open http://localhost:3000
```

### 6. Configure Flowise Chatflow

In the Flowise UI, build the following pipeline:

```
PDF File Loader
    ↓
Recursive Character Text Splitter (chunkSize: 1000, overlap: 200)
    ↓
OpenAI Embeddings (model: text-embedding-ada-002)
    ↓
Chroma (collection: scientists_kb, host: localhost:8000)
    ↓
ChatOpenAI (model: gpt-4o, temperature: 0)
    ↓
Conversational Retrieval QA Chain
```

1. Upload `Sample_Docs/scientists_knowledge_base.pdf` to the PDF Loader node
2. Select your OpenAI credential in the Embeddings node
3. Click **Upsert** — should show `Added: 10+` documents
4. Copy the Chatflow ID from the URL and set it in `.env`

---

## Running the Tests

### Run all tests
```bash
pytest . -v -s
```

### Run a specific metric test
```bash
pytest Faithfulness.py -v -s
pytest ContextPrecision.py -v -s
pytest MultipleMetrics.py -v -s
```

### Run by marker
```bash
pytest -m flowise -v -s     # All Flowise integration tests
pytest -m ragas -v -s       # All RAGAS metric tests
```

### Generate synthetic test data
```bash
python TestGen.py
# Saves testset to reports/testset_<timestamp>.csv and .json
```

---

## Test Results

`MultipleMetrics.py` saves results locally to `reports/`:

```
============================================================
RAGAS MULTI-METRIC EVALUATION RESULTS
============================================================
Metric                                    Score     Status
------------------------------------------------------------
factual_correctness                       0.8750    PASS
response_relevancy                        0.9123    PASS
faithfulness                              0.9500    PASS
llm_context_precision_without_reference  0.8800    PASS
context_recall                            0.7600    PASS
------------------------------------------------------------
Results saved to: reports/multi_metrics_2025-01-15_10-30-00.csv
```

---

## Key Design Decisions

**Why Flowise?**
Open-source, no-code RAG orchestration platform. Realistic enterprise-grade RAG pipeline without building from scratch. Exposes a REST API that can be tested like any other backend.

**Why RAGAS?**
Industry-standard framework for RAG evaluation. LLM-as-a-judge approach means no manual labeling — each metric has a clear mathematical definition and threshold.

**Why local reports instead of RAGAS dashboard?**
The RAGAS cloud dashboard (`app.ragas.io`) was deprecated. The framework saves results as CSV + JSON locally, making it CI/CD friendly (artifacts, Allure, etc.).

**Why `asyncio_mode=auto`?**
RAGAS scoring methods are async (`single_turn_ascore`). Setting `asyncio_mode=auto` in pytest.ini means no boilerplate `@pytest.mark.asyncio` needed on every test.

---

## Extending the Framework

- **Add a new metric**: Create a new file following the pattern in `Faithfulness.py`
- **Add CI/CD**: Use `pytest --junitxml=reports/results.xml` for Jenkins/GitHub Actions
- **Add Allure reporting**: `pip install allure-pytest` + `pytest --alluredir=reports/allure`
- **Custom knowledge base**: Replace PDFs in `Sample_Docs/` and update `TEST_DATA` in each test file

---

## Tech Stack

- **RAG Platform**: [Flowise](https://flowiseai.com/) (v3.x)
- **Vector DB**: [ChromaDB](https://www.trychroma.com/)
- **Embeddings + LLM**: OpenAI (`text-embedding-ada-002` + `gpt-4o`)
- **Evaluation**: [RAGAS](https://docs.ragas.io/) (v0.3.x)
- **Test Framework**: pytest + pytest-asyncio
- **Language**: Python 3.9+

---

## License

MIT
