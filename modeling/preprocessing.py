"""Phase3A: 前処理パイプライン(必ずTrainだけでfitする)。

sklearnのPipeline/ColumnTransformerを使い、前処理とモデルを一体化する。
数値特徴量: 中央値補完(0埋めは禁止)+ missing indicator列を追加。
availableフラグが既に存在する項目(turnover_ratio_available等)は、元のNULL値と
availableフラグの両方をそのまま数値特徴量として扱う(追加のmissing indicatorとは別)。
カテゴリ特徴量: OneHotEncoder(handle_unknown="ignore")で、Validationに未知カテゴリが
出てもエラーにならないようにする。

数値特徴量のみ(カテゴリ除外)と、数値+カテゴリの2パターンを比較できるよう
build_preprocessor(include_categorical=...)で切り替える。
"""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(numeric_cols: list[str], categorical_cols: list[str], include_categorical: bool) -> ColumnTransformer:
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
    ])
    transformers = [("numeric", numeric_pipeline, numeric_cols)]
    if include_categorical:
        categorical_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ])
        transformers.append(("categorical", categorical_pipeline, categorical_cols))
    return ColumnTransformer(transformers=transformers, remainder="drop")
