from openai import OpenAI
from ml_model import build_ml_result
from rag import retrieve_context

client = OpenAI()


def main():
    ml_result = build_ml_result()

    print("=== ML result ===")
    print(ml_result)

    query = f"""
以下は住宅価格予測モデルの結果です。
この会社の方針やヒアリング内容を踏まえて、
営業向けにどう説明すべきか整理してください。

特に以下を含めてください。
- このモデルで何が分かるか
- 営業現場でどう使うか
- 注意点
- 次アクション

モデル結果:
{ml_result}
""".strip()

    context, _ = retrieve_context(query=query, top_k=4, max_per_source=2)

    print("\n=== retrieved context ===")
    print(context)

    response = client.responses.create(
        model="gpt-5.4-mini",
        input=f"""
あなたは、不動産会社向けに分析結果を営業へ翻訳するアシスタントです。
以下の参考知識を必ず踏まえて回答してください。

参考知識:
{context}

以下は住宅価格予測モデルの結果です。
営業担当者向けに、専門用語をなるべく避けて説明してください。

含めてほしいこと:
1. このモデルで何が分かるか
2. 営業現場でどう使うとよいか
3. 注意点
4. 次にやるべきこと

モデル結果:
{ml_result}
"""
    )

    print("\n=== answer ===")
    print(response.output_text)


if __name__ == "__main__":
    main()