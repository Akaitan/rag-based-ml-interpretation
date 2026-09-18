import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional

from openai import OpenAI

from ml_model import build_ml_result
from rag import retrieve_context


BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_NAME = "gpt-5.4-mini"

client = OpenAI()


def build_user_prompt(ml_result: str) -> str:
    return f"""
以下は住宅価格予測モデルの結果です。
営業担当者向けに、専門用語をなるべく避けて説明してください。

必ず次の見出しで整理してください。
1. このモデルで何が分かるか
2. 営業現場でどう使うとよいか
3. 注意点
4. 次にやるべきこと

モデル結果:
{ml_result}
""".strip()


def generate_answer(user_prompt: str, context: Optional[str] = None) -> str:
    system_prompt = (
        "あなたは、不動産会社向けに分析結果を営業へ翻訳するアシスタントです。"
        "モデル結果から断定できないことは断定せず、根拠を明確にしてください。"
        "モデルを作ることが目的ではなく、成約を最大化するためにアシストしてください。"
        "社内の過去の失敗/成功例やルール、社内稟議などまで考慮したネクストアクションや注意点を洗い出してください。"
    )

    if context:
        system_prompt += f"""

以下の社内知識を業務固有の根拠として使ってください。
回答には、利用した情報の出典ファイル名を [source: ファイル名] の形で明記してください。
社内知識とモデル結果を混同しないでください。

社内知識:
{context}
"""
    else:
        system_prompt += (
            "社内知識は与えられていません。一般論を補う場合は、"
            "それが一般的な提案であることを明記してください。"
        )

    response = client.responses.create(
        model=MODEL_NAME,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.output_text


def build_report(ml_result: str, without_rag: str, with_rag: str, context: str) -> str:
    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    return f"""# RAGあり・なし比較レポート

- 生成日時: {generated_at}
- 生成モデル: {MODEL_NAME}
- 比較条件: モデル結果、質問、生成モデル、生成設定を共通化し、検索文脈の有無だけを変更

## ① モデル結果だけを生成AIに読ませた回答

{without_rag}

## ② RAGを使いつつモデル結果を生成AIに読ませた回答

{with_rag}

## RAGで取得した社内知識

{context}

## 比較するときの観点

| 観点 | 確認すること |
|---|---|
| 業務固有性 | 社内方針、営業ルール、ヒアリング内容が反映されたか |
| 根拠の追跡性 | 提案の根拠となる出典を確認できるか |
| 実行可能性 | 次アクションが具体的になったか |
| 正確性 | モデル結果や取得文書にない内容を断定していないか |
| 読みやすさ | 営業担当者が理解しやすい説明か |

## 共通で入力したモデル結果

```text
{ml_result}
```
"""


def main():
    parser = argparse.ArgumentParser(description="RAGあり・なしの回答を同条件で比較します。")
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DIR / "rag_comparison.md",
        help="比較レポートの保存先",
    )
    args = parser.parse_args()

    ml_result = build_ml_result()
    user_prompt = build_user_prompt(ml_result)
    context, selected = retrieve_context(query=user_prompt, top_k=4, max_per_source=2)

    print("① RAGなしの回答を生成しています...")
    without_rag = generate_answer(user_prompt)

    print("② RAGありの回答を生成しています...")
    with_rag = generate_answer(user_prompt, context=context)

    report = build_report(ml_result, without_rag, with_rag, context)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")

    print(f"\n比較レポートを保存しました: {args.output}")
    print("参照された文書:", ", ".join(sorted({item["source"] for item in selected})))


if __name__ == "__main__":
    main()
