

import argparse
import json
import os

from rag import load_vectorstore, build_qa_chain

EVAL_DIR = "eval"


def load_eval_set():
    with open(os.path.join(EVAL_DIR, "queries.json"), encoding="utf-8") as f:
        queries = json.load(f)
    with open(os.path.join(EVAL_DIR, "qrels.json"), encoding="utf-8") as f:
        qrels = json.load(f)
    with open(os.path.join(EVAL_DIR, "answers.json"), encoding="utf-8") as f:
        answers = json.load(f)
    return queries, qrels, answers


def word_overlap_score(predicted: str, reference: str) -> float:
    """A simple, dependency-free proxy metric: fraction of reference words
    that also appear in the predicted answer. Not a substitute for human
    judgment, but useful as a quick automated signal."""
    ref_words = set(reference.lower().split())
    pred_words = set(predicted.lower().split())
    if not ref_words:
        return 0.0
    return len(ref_words & pred_words) / len(ref_words)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                         help="Only evaluate the first N queries (default: all)")
    args = parser.parse_args()

    queries, qrels, answers = load_eval_set()

    if not queries:
        print("No queries found in eval/queries.json. Run download_ragbench.py first.")
        return

    query_ids = list(queries.keys())
    if args.limit:
        query_ids = query_ids[:args.limit]

    vectorstore = load_vectorstore()
    qa_chain = build_qa_chain(vectorstore)

    scores = []
    retrieval_hits = 0

    for qid in query_ids:
        question = queries[qid]["query"]
        reference_answer = answers.get(qid, "")
        expected_doc_id = qrels.get(qid, {}).get("doc_id")

        result = qa_chain.invoke({"query": question})
        predicted_answer = result["result"]
        retrieved_doc_ids = {
            os.path.splitext(os.path.basename(d.metadata.get("source", "")))[0]
            for d in result["source_documents"]
        }

        hit = expected_doc_id in retrieved_doc_ids
        retrieval_hits += int(hit)

        score = word_overlap_score(predicted_answer, reference_answer)
        scores.append(score)

        print("=" * 70)
        print(f"Query [{qid}] ({queries[qid].get('type', '?')}): {question}")
        print(f"Retrieved correct source doc: {'YES' if hit else 'no'}")
        print(f"Predicted answer:  {predicted_answer[:300]}")
        print(f"Reference answer:  {reference_answer[:300]}")
        print(f"Word-overlap score: {score:.2f}")

    n = len(query_ids)
    print("\n" + "=" * 70)
    print(f"SUMMARY over {n} queries")
    print(f"  Retrieval hit rate (correct source doc retrieved): {retrieval_hits}/{n} "
          f"({100*retrieval_hits/n:.1f}%)")
    print(f"  Avg word-overlap score vs reference answer: {sum(scores)/n:.2f}")
    print("\nNote: word-overlap is a rough proxy metric. For a more rigorous grade,")
    print("read through a few predicted vs reference answers manually.")


if __name__ == "__main__":
    main()
