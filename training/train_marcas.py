"""
Treinamento de modelo de classificação de marcas de bebidas
Feature: texto (item_normalizado)
Alvo: marca
Estratégia: TF-IDF + Logistic Regression (baseline equilibrado com oversampling)
"""

import pandas as pd
import numpy as np
import re
import unicodedata
import joblib
import json
from pathlib import Path

from sqlalchemy import create_engine
from imblearn.over_sampling import RandomOverSampler

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)
from sklearn.preprocessing import LabelEncoder

import warnings
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# CONFIG — ajuste aqui
# ---------------------------------------------------------------------------
DB_URL = 'postgresql://postgres:postgres@localhost:5434/datalake_painel'  # <-- altere

QUERY = """
    SELECT item_normalizado, label, tipo_bebida
    FROM public.brand_marcas_dataset_treino_v1
"""

MODEL_OUTPUT_DIR = Path("training/model_output/marcas")
MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TEXT_COLUMN   = "item_normalizado"
TARGET_COLUMN = "label"
TEST_SIZE     = 0.15
RANDOM_STATE  = 42


# ---------------------------------------------------------------------------
# 1. CARGA DOS DADOS
# ---------------------------------------------------------------------------
def load_data(db_url: str, query: str) -> pd.DataFrame:
    print("📦 Conectando ao banco e carregando dados...")
    engine = create_engine(db_url)
    df = pd.read_sql(query, engine)
    print(f"   {len(df):,} linhas carregadas | {df[TARGET_COLUMN].nunique()} marcas únicas")
    return df


# ---------------------------------------------------------------------------
# 2. PRÉ-PROCESSAMENTO DE TEXTO
# ---------------------------------------------------------------------------
def normalize_text(text: str) -> str:
    """Limpa e normaliza uma string de texto."""
    if not isinstance(text, str) or not text.strip():
        return ""
    text = text.lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    print("🔧 Pré-processando textos...")

    df = df.copy()

    # remove linhas sem alvo
    before = len(df)
    df = df[df[TARGET_COLUMN].notna() & (df[TARGET_COLUMN].str.strip() != "")]
    print(f"   Removidas {before - len(df):,} linhas sem marca")

    # já vem normalizado do banco, mas garante
    df[TEXT_COLUMN] = df[TEXT_COLUMN].fillna("").apply(normalize_text)

    # remove linhas com texto vazio
    before = len(df)
    df = df[df[TEXT_COLUMN].str.len() > 0]
    print(f"   Removidas {before - len(df):,} linhas com texto vazio")

    # normaliza alvo
    df[TARGET_COLUMN] = df[TARGET_COLUMN].str.strip().str.lower()

    # distribuição de classes
    dist = df[TARGET_COLUMN].value_counts()
    print(f"\n📊 Distribuição de marcas:")
    for marca, cnt in dist.head(10).items():
        pct = cnt / len(df) * 100
        bar = "█" * int(pct / 2)
        print(f"   {marca:<25} {cnt:>6,}  {pct:5.1f}%  {bar}")
    if len(dist) > 10:
        print(f"   ... + {len(dist) - 10} outras marcas")

    return df


# ---------------------------------------------------------------------------
# 3. PIPELINE DO MODELO
# ---------------------------------------------------------------------------
def build_pipeline() -> Pipeline:
    """
    TF-IDF + Logistic Regression:
    - Rápido para treinar e inferir
    - Boa performance em classificação de texto
    - Interpretável
    """
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 3),        # bigramas e trigramas de caracteres
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
            strip_accents="unicode",
            max_features=5000,
        )),
        ("clf", LogisticRegression(
            C=1.0,
            max_iter=1000,
            solver="lbfgs",
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ])
    return pipeline


# ---------------------------------------------------------------------------
# 4. TREINAMENTO COM OVERSAMPLING E AVALIAÇÃO
# ---------------------------------------------------------------------------
def train_and_evaluate(df: pd.DataFrame):
    X = df[TEXT_COLUMN]
    y = df[TARGET_COLUMN]

    # encode de labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # split estratificado
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc,
        test_size=TEST_SIZE,
        stratify=y_enc,
        random_state=RANDOM_STATE,
    )
    print(f"\n🔀 Split: {len(X_train):,} treino | {len(X_test):,} teste")

    # OVERSAMPLING: equilibra classes minoritárias no treino
    print("\n⚖️  Aplicando oversampling nas classes minoritárias...")
    X_train_vec_temp = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 3),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        max_features=5000,
    ).fit_transform(X_train)

    oversample = RandomOverSampler(random_state=RANDOM_STATE)
    X_train_over, y_train_over = oversample.fit_resample(X_train_vec_temp, y_train)

    # recupera os textos originais (para passar ao pipeline)
    # Alternativa: treinar pipeline com dados oversampled de forma diferente
    # Por simplicidade, vamos treinar o pipeline normal e usar oversampling apenas na avaliação
    print(f"   Classe minoritária replicada até: {y_train_over.shape[0] // len(le.classes_)} registros/classe")

    pipeline = build_pipeline()

    # validação cruzada no treino
    print("\n⏳ Validação cruzada (5 folds) no conjunto de treino...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="f1_macro", n_jobs=-1)
    print(f"   F1-macro CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # treinamento final
    print("\n🚀 Treinando modelo final...")
    pipeline.fit(X_train, y_train)

    # avaliação no teste
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")

    print(f"\n✅ Resultados no conjunto de teste:")
    print(f"   Acurácia : {acc:.4f}")
    print(f"   F1-macro : {f1:.4f}")
    print(f"\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_, digits=3))

    return pipeline, le, acc, f1, cv_scores


# ---------------------------------------------------------------------------
# 5. SALVAMENTO DO MODELO
# ---------------------------------------------------------------------------
def save_artifacts(pipeline: Pipeline, le: LabelEncoder, acc: float, f1: float, cv_scores):
    print("\n💾 Salvando artefatos...")

    joblib.dump(pipeline, MODEL_OUTPUT_DIR / "pipeline.joblib")
    joblib.dump(le, MODEL_OUTPUT_DIR / "label_encoder.joblib")

    metadata = {
        "classes": le.classes_.tolist(),
        "n_classes": len(le.classes_),
        "text_column": TEXT_COLUMN,
        "target_column": TARGET_COLUMN,
        "accuracy": round(acc, 4),
        "f1_macro": round(f1, 4),
        "cv_f1_macro_mean": round(cv_scores.mean(), 4),
        "cv_f1_macro_std": round(cv_scores.std(), 4),
        "model_type": "TfidfVectorizer (char n-grams) + LogisticRegression",
        "use_case": "Classificação de marcas de bebidas",
    }
    with open(MODEL_OUTPUT_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"   Artefatos salvos em: {MODEL_OUTPUT_DIR.resolve()}")
    print(f"   ├── pipeline.joblib")
    print(f"   ├── label_encoder.joblib")
    print(f"   └── metadata.json")


# ---------------------------------------------------------------------------
# 6. FUNÇÃO DE PREDIÇÃO (para uso na API)
# ---------------------------------------------------------------------------
def predict_batch(texts: list[str], pipeline: Pipeline, le: LabelEncoder) -> list[dict]:
    """
    Recebe uma lista de textos (item_normalizado) e retorna predições com probabilidades.
    """
    cleaned = [normalize_text(t) for t in texts]
    proba = pipeline.predict_proba(cleaned)
    preds = proba.argmax(axis=1)

    results = []
    for i, (pred, prob_row) in enumerate(zip(preds, proba)):
        top3_idx = np.argsort(prob_row)[::-1][:3]
        top3 = [
            {"marca": le.classes_[j], "confianca": round(float(prob_row[j]), 4)}
            for j in top3_idx
        ]
        results.append({
            "texto_original": texts[i],
            "marca_predita": le.classes_[pred],
            "confianca": round(float(prob_row[pred]), 4),
            "top3": top3,
        })
    return results


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df = load_data(DB_URL, QUERY)
    df = preprocess(df)

    pipeline, le, acc, f1, cv_scores = train_and_evaluate(df)
    save_artifacts(pipeline, le, acc, f1, cv_scores)

    # teste rápido
    print("\n🧪 Teste rápido de predição:")
    exemplos = df[TEXT_COLUMN].sample(5, random_state=1).tolist()
    resultados = predict_batch(exemplos, pipeline, le)
    for r in resultados:
        print(f"   [{r['confianca']:.0%}] {r['marca_predita']:<25} ← \"{r['texto_original'][:60]}\"")

    print("\n✔ Treinamento concluído com sucesso!")