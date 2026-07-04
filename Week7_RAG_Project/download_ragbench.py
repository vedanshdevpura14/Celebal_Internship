"""
Download & prepare the Open RAG Benchmark dataset (vectara/open_ragbench)
==========================================================================
This dataset is NOT a simple CSV/JSON table - it's a BEIR-style benchmark
built from arXiv PDFs, structured as:

    official/pdf/arxiv/
    ├── corpus/{PAPER_ID}.json   -> title, abstract, sections (text+tables+images)
    ├── queries.json             -> ~3045 questions
    ├── qrels.json               -> which paper/section answers which query
    ├── answers.json             -> ground-truth answers
    └── pdf_urls.json

This script:
  1. Downloads a sample of N papers from `corpus/`
  2. Converts each paper's sections into a plain-text file (for your RAG pipeline)
  3. Filters queries.json / qrels.json / answers.json down to only the
     questions that belong to the papers you downloaded
  4. Saves everything into data/ and eval/ so rag.py can use it directly

USAGE
-----
    pip install huggingface_hub
    python download_ragbench.py --num-papers 20

Then run your existing pipeline:
    python rag.py ingest
    python rag.py ask "some question"

Or evaluate against ground truth:
    python evaluate.py
"""

import argparse
import json
import os
import random

from huggingface_hub import HfApi, hf_hub_download

REPO_ID = "vectara/open_ragbench"
REPO_TYPE = "dataset"
BASE_PATH = "official/pdf/arxiv"

DATA_DIR = "data"
EVAL_DIR = "eval"


def paper_json_to_text(paper: dict) -> str:
    """Flatten a corpus paper JSON into plain text (title, abstract, sections).
    Tables are kept as markdown; images are replaced with a short placeholder
    since we can't feed base64 images into a text-only pipeline here."""
    lines = []
    lines.append(f"Title: {paper.get('title', '')}")
    if paper.get("authors"):
        lines.append("Authors: " + ", ".join(paper["authors"]))
    if paper.get("categories"):
        lines.append("Categories: " + ", ".join(paper["categories"]))
    lines.append("")
    lines.append("Abstract:")
    lines.append(paper.get("abstract", ""))
    lines.append("")

    for i, section in enumerate(paper.get("sections", [])):
        text = section.get("text", "")
        tables = section.get("tables", {})
        for table_id, table_md in tables.items():
            # Replace table placeholder tokens with the actual markdown table
            text = text.replace(f"[{table_id}]", f"\n{table_md}\n")
        if section.get("images"):
            text += "\n[This section also contains image(s), not included in this text export.]"
        lines.append(f"--- Section {i+1} ---")
        lines.append(text)
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-papers", type=int, default=20,
                         help="How many papers to download (dataset has 1000 total, ~700MB+)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(EVAL_DIR, exist_ok=True)

    api = HfApi()

    print("Listing files in dataset repo (this may take a moment)...")
    all_files = api.list_repo_files(repo_id=REPO_ID, repo_type=REPO_TYPE)
    corpus_files = [f for f in all_files if f.startswith(f"{BASE_PATH}/corpus/") and f.endswith(".json")]
    print(f"Found {len(corpus_files)} papers in corpus/.")

    random.seed(args.seed)
    sample = random.sample(corpus_files, min(args.num_papers, len(corpus_files)))
    print(f"Downloading {len(sample)} papers...")

    downloaded_ids = []
    for f in sample:
        local_path = hf_hub_download(repo_id=REPO_ID, repo_type=REPO_TYPE, filename=f)
        with open(local_path, "r", encoding="utf-8") as fh:
            paper = json.load(fh)

        paper_id = paper.get("id") or os.path.splitext(os.path.basename(f))[0]
        downloaded_ids.append(paper_id)

        text = paper_json_to_text(paper)
        out_path = os.path.join(DATA_DIR, f"{paper_id}.txt")
        with open(out_path, "w", encoding="utf-8") as out:
            out.write(text)
        print(f"  wrote {out_path}")

    print("\nDownloading queries.json, qrels.json, answers.json ...")
    queries_path = hf_hub_download(repo_id=REPO_ID, repo_type=REPO_TYPE,
                                    filename=f"{BASE_PATH}/queries.json")
    qrels_path = hf_hub_download(repo_id=REPO_ID, repo_type=REPO_TYPE,
                                  filename=f"{BASE_PATH}/qrels.json")
    answers_path = hf_hub_download(repo_id=REPO_ID, repo_type=REPO_TYPE,
                                    filename=f"{BASE_PATH}/answers.json")

    with open(queries_path, encoding="utf-8") as f:
        all_queries = json.load(f)
    with open(qrels_path, encoding="utf-8") as f:
        all_qrels = json.load(f)
    with open(answers_path, encoding="utf-8") as f:
        all_answers = json.load(f)

    downloaded_ids_set = set(downloaded_ids)

    # Keep only queries whose qrel points to a paper we actually downloaded
    filtered_qrels = {qid: qr for qid, qr in all_qrels.items() if qr.get("doc_id") in downloaded_ids_set}
    filtered_queries = {qid: all_queries[qid] for qid in filtered_qrels if qid in all_queries}
    filtered_answers = {qid: all_answers[qid] for qid in filtered_qrels if qid in all_answers}

    with open(os.path.join(EVAL_DIR, "queries.json"), "w", encoding="utf-8") as f:
        json.dump(filtered_queries, f, indent=2)
    with open(os.path.join(EVAL_DIR, "qrels.json"), "w", encoding="utf-8") as f:
        json.dump(filtered_qrels, f, indent=2)
    with open(os.path.join(EVAL_DIR, "answers.json"), "w", encoding="utf-8") as f:
        json.dump(filtered_answers, f, indent=2)

    print(f"\nDone!")
    print(f"  {len(downloaded_ids)} papers -> {DATA_DIR}/")
    print(f"  {len(filtered_queries)} matching queries -> {EVAL_DIR}/queries.json")
    print(f"  Ground-truth answers -> {EVAL_DIR}/answers.json")
    print("\nNext steps:")
    print("  python rag.py ingest")
    print("  python evaluate.py")


if __name__ == "__main__":
    main()
