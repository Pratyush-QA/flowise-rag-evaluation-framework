import os
import csv
import json
import nltk
import pytest
from datetime import datetime
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.testset import TestsetGenerator

load_dotenv()

# -------------------------------------------------------------------
# TEST DATA GENERATION — SAVED LOCALLY
# Auto-generates test questions & answers from your knowledge base docs.
# Results saved locally to reports/ folder as CSV and JSON.
#
# How it works:
#   - Loads markdown/text documents from Sample_Docs folder
#   - RAGAS TestsetGenerator creates realistic Q&A pairs
#   - Generated test cases saved to reports/testset_<timestamp>.csv
#   - Use this generated data as test inputs for metric evaluations
# -------------------------------------------------------------------

REPORTS_DIR = os.path.join(os.path.dirname(__file__), "reports")


@pytest.mark.ragas
def test_data_creation():
    base_dir = os.path.dirname(__file__)
    docs_path = os.path.join(base_dir, "Sample_Docs")

    if not os.path.exists(docs_path) or not os.listdir(docs_path):
        pytest.skip(
            f"No documents found in {docs_path}\n"
            "Add .md or .txt files to Sample_Docs/ folder to generate test data."
        )

    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    llm_ragas = LangchainLLMWrapper(llm)

    embeddings_openai = OpenAIEmbeddings()
    ragas_embeddings = LangchainEmbeddingsWrapper(embeddings_openai)

    loader = DirectoryLoader(docs_path, glob="**/*.md")
    docs = loader.load()

    print(f"\nLoaded {len(docs)} document(s) from {docs_path}")

    testgen = TestsetGenerator(llm=llm_ragas, embedding_model=ragas_embeddings)
    dataset = testgen.generate_with_langchain_docs(docs, testset_size=10)

    generated = dataset.to_list()
    print(f"\nGenerated {len(generated)} test samples:")
    for i, item in enumerate(generated[:3], 1):
        print(f"  [{i}] Q: {item.get('user_input', '')[:80]}...")

    # ── Save generated test set locally ──────────────────────────
    os.makedirs(REPORTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. Save as CSV
    csv_path = os.path.join(REPORTS_DIR, f"testset_{timestamp}.csv")
    if generated:
        fieldnames = list(generated[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(generated)
        print(f"\n[TestGen] CSV saved → {csv_path}")

    # 2. Save as JSON
    json_path = os.path.join(REPORTS_DIR, f"testset_{timestamp}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(generated, f, indent=2, ensure_ascii=False)
    print(f"[TestGen] JSON saved → {json_path}")

    # 3. Print all generated Q&A pairs
    print(f"\n{'='*60}")
    print(f"Generated Test Cases")
    print(f"{'='*60}")
    for i, item in enumerate(generated, 1):
        print(f"\n[{i}] Question : {item.get('user_input', 'N/A')}")
        print(f"     Reference: {str(item.get('reference', 'N/A'))[:120]}")
        print(f"     Type     : {item.get('synthesizer_name', 'N/A')}")

    assert len(generated) > 0, "No test data was generated — check your documents."


if __name__ == "__main__":
    test_data_creation()
