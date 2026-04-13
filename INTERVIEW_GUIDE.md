# Interview Guide — Flowise RAG Evaluation Framework

_For Senior SDET / Lead SDET roles discussing this project_

---

## The 60-Second Project Pitch

> "I built an end-to-end RAG evaluation framework using Flowise and RAGAS. The idea came from a real gap — most QA engineers can test APIs and UIs, but AI systems need a completely different evaluation strategy because there's no deterministic output. I set up a real Flowise RAG pipeline backed by ChromaDB and OpenAI, then wrote a pytest-based suite that evaluates 7 RAGAS metrics — things like faithfulness (hallucination detection), factual correctness, and context precision. Each test makes a real HTTP call to the Flowise prediction API, gets back the LLM response plus source documents, feeds that into RAGAS, and asserts against threshold scores. Results are saved as CSV and JSON for CI/CD integration."

---

## Why This Project Matters (QA Angle)

**Traditional QA challenge**: LLM responses are non-deterministic. You can't assert `response == expected`.

**This framework solves it by**:
- Using LLM-as-a-judge (RAGAS) to score responses against dimensions like faithfulness and relevancy
- Setting threshold-based assertions (e.g., faithfulness >= 0.8) — the same pattern as performance testing
- Testing the *entire RAG pipeline* end-to-end, not just the LLM in isolation

---

## Common Interview Questions and Answers

---

### "Walk me through your test architecture."

The framework has four layers:

1. **RAG Pipeline** (Flowise): PDF → Text Splitter → OpenAI Embeddings → ChromaDB → GPT-4o → Conversational Retrieval Chain
2. **API Client** (`flowise_client.py`): Wraps the Flowise `/prediction` endpoint, extracts both the response text and `sourceDocuments` (retrieved contexts)
3. **Shared Fixtures** (`conftest.py`): `llm_wrapper` (LangchainLLMWrapper around GPT-4o) and `flowise_chatflow_id` from `.env` — no hardcoded keys
4. **Metric Tests**: Each file tests one RAGAS metric, creates a `SingleTurnSample`, scores it, and asserts against a threshold

---

### "Why did you choose RAGAS over other evaluation frameworks?"

RAGAS is the industry standard for RAG evaluation. Key reasons:
- **Reference-free metrics**: Faithfulness and Response Relevancy don't need ground truth — they use the retrieved context and query alone
- **Reference-based metrics**: Factual Correctness and Context Recall compare against ground truth answers for deeper validation
- **Interpretable scores**: Each metric is 0-1 (or 1-5 for Rubric), with clear mathematical definitions
- Alternatives like DeepEval and TruLens are valid, but RAGAS has stronger community adoption and direct RAG focus

---

### "What is Faithfulness and why is your threshold higher (0.8 vs 0.7 for others)?"

Faithfulness measures whether every claim in the LLM response is supported by the retrieved context. A low score = hallucination — the model made up facts not in the documents.

I set it to 0.8 (stricter) because hallucinations are the most dangerous failure mode in a RAG system. If the system confidently answers with wrong information, that's a P1 defect in a production AI product. A business can tolerate a slightly incomplete answer; it cannot tolerate a factually fabricated one.

---

### "How does the Flowise client work? What does it return?"

```python
result = query_flowise(question=query, chatflow_id=chatflow_id)
# result["response"]          — the LLM-generated answer text
# result["retrieved_contexts"] — list of raw text chunks from ChromaDB
```

The Flowise `/prediction` API returns a JSON with `text` (the answer) and `sourceDocuments` (the retrieved chunks). I extract both because RAGAS needs the retrieved contexts to evaluate faithfulness and context precision — not just the final answer.

---

### "What's the difference between Context Precision and Context Recall?"

| Metric | Analogy | Question it answers |
|---|---|---|
| **Context Precision** | Search Precision | Of everything retrieved, how much was actually relevant? |
| **Context Recall** | Search Recall | Of all relevant info needed, how much did we actually retrieve? |

Together they tell you about your retrieval quality. Poor precision = too much noise retrieved. Poor recall = missing key information, leading to incomplete answers.

---

### "How does Topic Adherence work? What's the multi-turn angle?"

Topic Adherence uses Flowise's `sessionId` to maintain conversation state across multiple turns. I send a sequence of related questions in a single session and check that the LLM stays on-topic throughout. This tests the retrieval chain's memory component and validates that context isn't corrupted or lost across turns.

---

### "What is RubricsScore and when would you use it?"

RubricsScore is a custom 1-5 evaluator where you define what each score means for your domain. For example:
- Score 1: completely wrong
- Score 5: fully accurate, detailed, aligned with reference

This is the most flexible metric — ideal when stakeholders want a subjective quality bar ("is this a good answer for a customer support bot?") that you can calibrate for your specific use case. I set the threshold at >= 4.

---

### "Why are you saving reports locally instead of using the RAGAS dashboard?"

The RAGAS cloud dashboard (`app.ragas.io`) was deprecated — the service went offline. This was actually a good learning: **don't build critical test infrastructure on third-party SaaS without an offline fallback**.

I refactored `MultipleMetrics.py` to save results as CSV and JSON locally with timestamps. This is more CI/CD friendly anyway — you can publish artifacts in GitHub Actions, index them in Elasticsearch, or serve them through Allure without any external dependency.

---

### "How would you run this in CI/CD?"

```yaml
# GitHub Actions example
- name: Start ChromaDB
  run: chroma run --host localhost --port 8000 &

- name: Start Flowise
  run: flowise start &

- name: Wait for services
  run: sleep 10

- name: Run RAGAS evaluation
  run: pytest . -v --junitxml=reports/results.xml
  env:
    OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
    FLOWISE_CHATFLOW_ID: ${{ secrets.FLOWISE_CHATFLOW_ID }}

- name: Upload results
  uses: actions/upload-artifact@v3
  with:
    name: ragas-results
    path: reports/
```

For cost control in CI, I'd gate the RAGAS tests behind a `@pytest.mark.ragas` marker and only run them on PR to main, not on every commit.

---

### "What would you do differently in a real enterprise project?"

1. **Parameterized test data from external source**: Instead of hardcoded `TEST_DATA` lists, load from a YAML or CSV maintained by the team
2. **Score trending**: Store results in a database (PostgreSQL/InfluxDB) and alert on score regression — if faithfulness drops from 0.92 to 0.75 between releases, fail the build
3. **Separate evaluation LLM from the tested LLM**: Currently both use GPT-4o. In production, I'd use a separate, cheaper model (e.g., GPT-4o-mini) as the RAGAS judge to reduce cost
4. **Contract testing for the Flowise API**: Add schema validation for the `/prediction` response structure using `pydantic` or `jsonschema`
5. **Parallel test execution**: Use `pytest-xdist` to run metric tests in parallel since each is independent

---

### "What is RAG and how does it work?"

RAG = Retrieval-Augmented Generation.

**Problem it solves**: LLMs have a knowledge cutoff and can't answer questions about private/proprietary documents.

**How it works**:
1. **Ingestion**: Documents are split into chunks, converted to vector embeddings, stored in a vector DB (ChromaDB)
2. **Retrieval**: User query is embedded, and semantically similar chunks are retrieved from the DB
3. **Generation**: Retrieved chunks + user query are injected into the LLM prompt as context; LLM generates a grounded answer

**Why QA cares**: The LLM response is only as good as what was retrieved. Testing the retrieval quality (precision, recall) is as important as testing the LLM output quality.

---

### "What bugs or issues did you encounter while building this?"

**Flowise ChromaDB connection error**: Flowise returned `ChromaConnectionError` because I hadn't started the ChromaDB server. Fixed with `chroma run --host localhost --port 8000`. Added this to the README as a prerequisite.

**Zero documents upserted**: The OpenAI Embeddings node in Flowise had no credential selected — a UI misconfiguration. The upsert returned `Added: 0`. Taught me to always validate preconditions before running tests (data is in the vector DB).

**RAGAS dashboard deprecated**: `upload()` method and `app.ragas.io` went offline. Refactored to local reports — a good reminder to keep external dependencies to a minimum in test infrastructure.

**RubricScore assertion bug**: Original code had `assert score > 6` on a 1-5 scale. This always fails — classic off-by-one combined with wrong range assumption. Fixed to `assert score >= 4`.

---

### "How is this different from just calling OpenAI directly and checking the output?"

Calling OpenAI directly tests the LLM in isolation. This framework tests the **entire pipeline**:

- Is the retrieval finding the right chunks? (Context Precision/Recall)
- Is the LLM staying faithful to those chunks? (Faithfulness)
- Is the final answer factually correct vs our known truth? (Factual Correctness)
- Is the answer actually relevant to the question? (Response Relevancy)

A RAG system can fail at any of these layers. You need metrics for each layer.

---

## Resume Bullet Points

```
• Built a RAGAS-based evaluation framework for Flowise RAG pipelines, covering 7 quality 
  metrics including faithfulness (hallucination detection), factual correctness, and context 
  precision/recall using pytest + pytest-asyncio

• Designed threshold-based assertions (faithfulness >= 0.8, others >= 0.7) with local 
  CSV/JSON reporting — CI/CD ready, no external dashboard dependency

• Integrated real Flowise HTTP prediction API calls into test fixtures via shared conftest.py, 
  decoupling LLM configuration from test logic; zero hardcoded API keys

• Configured end-to-end RAG pipeline: PDF Loader → Text Splitter → OpenAI Embeddings → 
  ChromaDB → GPT-4o → Conversational Retrieval Chain in Flowise 3.x
```

---

## Tech Stack Summary Card

| Layer | Technology | Why |
|---|---|---|
| RAG Platform | Flowise 3.x | Real open-source RAG, REST API |
| Vector DB | ChromaDB | Local, lightweight, realistic |
| Embeddings | OpenAI text-embedding-ada-002 | Industry standard |
| LLM | GPT-4o | Best accuracy for evaluation |
| Evaluation | RAGAS 0.3.x | RAG-specific metrics, LLM-as-judge |
| Test Framework | pytest + pytest-asyncio | Async support, markers, fixtures |
| Reports | CSV + JSON (local) | CI/CD friendly, no external SaaS |
