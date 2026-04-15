import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from lightgbm import LGBMRegressor


def build_ml_result():
    housing = fetch_california_housing(as_frame=True)
    X = housing.data
    y = housing.target  # 10万ドル単位

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=-1,
        num_leaves=31,
        random_state=42
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2 = r2_score(y_test, y_pred)

    feature_importance = pd.DataFrame({
        "feature": X.columns,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    top_features = feature_importance.head(5)

    feature_text = "\n".join(
        f"- {row.feature}: {row.importance}"
        for _, row in top_features.iterrows()
    )

    sample_df = pd.DataFrame({
        "actual_price_100k": y_test.iloc[:5].values,
        "predicted_price_100k": y_pred[:5]
    })

    sample_text = "\n".join(
        f"- 実測: {row.actual_price_100k:.2f}, 予測: {row.predicted_price_100k:.2f}"
        for _, row in sample_df.iterrows()
    )

    ml_result = f"""
モデル: LightGBM Regressor
目的: California Housing データセットを使った住宅価格予測
目的変数の単位: 10万ドル

データ件数:
- 学習データ: {len(X_train)}
- テストデータ: {len(X_test)}

評価結果:
- MAE: {mae:.3f}
- RMSE: {rmse:.3f}
- R2: {r2:.3f}

重要特徴量 上位5件:
{feature_text}

予測例:
{sample_text}
""".strip()

    return ml_result