import json
import math
from pathlib import Path
from openai import OpenAI

client = OpenAI()

BASE_DIR = Path(__file__).resolve().parent.parent
INDEX_FILE = BASE_DIR / "data" / "index.json"


def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b)


def load_index():
    return json.loads(INDEX_FILE.read_text(encoding="utf-8"))


def retrieve_context(query: str, top_k: int = 4, max_per_source: int = 2):
    chunks = load_index()

    query_emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    ).data[0].embedding

    scored = []
    for chunk in chunks:
        score = cosine_similarity(query_emb, chunk["embedding"])
        scored.append((score, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)

    selected = []
    used_sources = {}

    for _, chunk in scored:
        source = chunk["source"]
        used_sources[source] = used_sources.get(source, 0)

        if used_sources[source] >= max_per_source:
            continue

        selected.append(chunk)
        used_sources[source] += 1

        if len(selected) >= top_k:
            break

    context = "\n\n".join(
        f"[source: {c['source']}]\n{c['text']}" for c in selected
    )

    return context, selected