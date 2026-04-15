from pathlib import Path
import json
from openai import OpenAI

client = OpenAI()

BASE_DIR = Path(__file__).resolve().parent.parent
KNOWLEDGE_DIR = BASE_DIR / "knowledge"
OUTPUT_FILE = BASE_DIR / "data" / "index.json"


def load_documents():
    docs = []
    for path in KNOWLEDGE_DIR.glob("*.md"):
        text = path.read_text(encoding="utf-8").strip()
        if text:
            docs.append({
                "source": path.name,
                "text": text
            })
    return docs


def simple_chunk(text: str, max_chars: int = 400):
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []

    for para in paragraphs:
        # 段落が短ければそのまま1チャンク
        if len(para) <= max_chars:
            chunks.append(para)
        else:
            # 長すぎる段落だけ、句点ベースでゆるく分割
            sentences = para.split("。")
            current = ""

            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue

                candidate = sentence + "。"

                if len(current) + len(candidate) <= max_chars:
                    current += candidate
                else:
                    if current:
                        chunks.append(current.strip())
                    current = candidate

            if current:
                chunks.append(current.strip())

    return chunks


docs = load_documents()

chunks = []
for doc in docs:
    for chunk in simple_chunk(doc["text"]):
        chunks.append({
            "source": doc["source"],
            "text": chunk
        })

texts = [c["text"] for c in chunks]

emb = client.embeddings.create(
    model="text-embedding-3-small",
    input=texts
)

for chunk, item in zip(chunks, emb.data):
    chunk["embedding"] = item.embedding

OUTPUT_FILE.write_text(
    json.dumps(chunks, ensure_ascii=False),
    encoding="utf-8"
)

print(f"saved {len(chunks)} chunks to {OUTPUT_FILE}")