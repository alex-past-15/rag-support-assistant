"""Build a local semantic index and retrieve documentation fragments."""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "data/index"
MODEL = "intfloat/multilingual-e5-small"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def corpus_hash():
    digest = hashlib.sha256()
    digest.update((ROOT / "data/sources.json").read_bytes())
    for path in sorted((ROOT / "data/processed").glob("*.json")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def sections(text):
    # The footer is part of every article and adds irrelevant matches.
    text = re.split(r"(?m)^## Полезные ссылки\s*$", text)[0]
    text = re.sub(r"!\[[^\]]*\]\([^\n]*?\)", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^\n]*?\)", r"\1", text)
    headings, lines = [], []
    for line in text.splitlines():
        match = re.match(r"^(#{1,6})\s+(.+)$", line)
        if match:
            if "\n".join(lines).strip():
                yield " / ".join(t for _, t in headings), "\n".join(lines).strip()
            level = len(match[1])
            headings = [(n, t) for n, t in headings if n < level]
            headings.append((level, match[2]))
            lines = []
        else:
            lines.append(line)
    if "\n".join(lines).strip():
        yield " / ".join(t for _, t in headings), "\n".join(lines).strip()


def make_chunks(doc, tokenizer):
    result = []
    for heading, text in sections(doc["text"]):
        prefix = f"passage: {doc['title']}\n{heading}\n"
        budget = 480 - len(tokenizer.encode(prefix, add_special_tokens=True))
        if budget < 80:
            raise ValueError("Heading too long for model context")
        offsets = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True,
                            truncation=False, verbose=False)["offset_mapping"]
        start = 0
        while start < len(offsets):
            end = min(start + budget, len(offsets))
            # Prefer a paragraph boundary in the second half of the window.
            if end < len(offsets):
                boundary = text.rfind("\n\n", offsets[start + budget // 2][0], offsets[end - 1][1])
                if boundary >= 0:
                    while end > start and offsets[end - 1][0] >= boundary:
                        end -= 1
            fragment = text[offsets[start][0]:offsets[end - 1][1]].strip()
            embedded = prefix + fragment
            if len(tokenizer.encode(embedded)) > 512:
                raise ValueError("Chunk exceeds model context")
            result.append({"chunk_id": f"{doc['id']}_{len(result)+1:03d}",
                           "doc_id": doc["id"], "title": doc["title"],
                           "section": heading, "text": fragment,
                           "source_url": doc["source_url"],
                           "downloaded_at": doc["downloaded_at"],
                           "needs_image_review": doc["needs_image_review"],
                           "embedding_text": embedded})
            if end == len(offsets):
                break
            start = max(start + 1, end - 40)
    return result


def build(model):
    sources = read_json(ROOT / "data/sources.json")
    documents = [read_json(ROOT / f"data/processed/{s['id']}.json") for s in sources]
    if any(d["source_url"] != s["url"] for d, s in zip(documents, sources)):
        raise ValueError("Source list differs from processed documents")
    chunks = [c for doc in documents for c in make_chunks(doc, model.tokenizer)]
    if not chunks:
        raise ValueError("No chunks to index")
    vectors = model.encode([c["embedding_text"] for c in chunks],
                           normalize_embeddings=True, show_progress_bar=True, batch_size=16)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors.astype("float32"))
    INDEX.mkdir(parents=True, exist_ok=True)
    # Byte serialization also supports Windows paths containing Cyrillic.
    (INDEX / "vectors.faiss").write_bytes(faiss.serialize_index(index).tobytes())
    (INDEX / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    (INDEX / "metadata.json").write_text(json.dumps({"model": MODEL, "corpus_hash": corpus_hash(),
        "documents": len(documents), "chunks": len(chunks)}, indent=2), encoding="utf-8")
    print(f"Indexed {len(documents)} documents, {len(chunks)} chunks")


def load_index():
    import numpy as np
    meta = read_json(INDEX / "metadata.json")
    if meta["model"] != MODEL or meta["corpus_hash"] != corpus_hash():
        raise ValueError("Index is stale. Run build again.")
    index = faiss.deserialize_index(np.frombuffer((INDEX / "vectors.faiss").read_bytes(), dtype="uint8"))
    chunks = read_json(INDEX / "chunks.json")
    if index.ntotal != len(chunks):
        raise ValueError("Index and chunk count differ")
    return index, chunks


def search(model, index, chunks, question, k):
    if not question.strip() or k < 1:
        raise ValueError("Enter a non-empty question and positive k")
    if len(model.tokenizer.encode("query: " + question)) > 512:
        raise ValueError("Question is too long; shorten it")
    vector = model.encode(["query: " + question], normalize_embeddings=True)
    scores, ids = index.search(vector.astype("float32"), min(k, len(chunks)))
    return [{**chunks[i], "score": float(score)} for i, score in zip(ids[0], scores[0])]


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["build", "search", "evaluate"])
    parser.add_argument("question", nargs="?")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()
    model = SentenceTransformer(MODEL, device="cpu")
    model.max_seq_length = 512
    if args.action == "build":
        build(model)
        return
    index, chunks = load_index()
    if args.action == "search":
        if not args.question:
            parser.error("search requires a question")
        for row in search(model, index, chunks, args.question, args.top_k):
            print(f"\n{row['score']:.3f} | {row['title']} | {row['section']}\n{row['source_url']}\n{row['text']}\n")
        print("Score is similarity, not confidence. Results are source excerpts, not an answer.")
    else:
        cases = read_json(ROOT / "data/search_checks.json")
        results = []
        for case in cases:
            hits = search(model, index, chunks, case["question"], 3)
            retrieved = [h["doc_id"] for h in hits]
            results.append({**case, "retrieved": retrieved,
                            "hit_at_3": case["expected_doc"] in retrieved})
        report = {"note": "Development smoke check, not an independent quality evaluation",
                  "hit_at_3": sum(r["hit_at_3"] for r in results) / len(results), "cases": results}
        (INDEX / "search_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
