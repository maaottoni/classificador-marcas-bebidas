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
    accuracy_score,
    f1_score,
)
from sklearn.preprocessing import LabelEncoder

import warnings
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# CONFIG — ajuste aqui
# ---------------------------------------------------------------------------
DB_URL = 'postgresql://postgres:postgres@localhost:5434/postgres'

QUERY = """
    SELECT item_normalizado, label, tipo_bebida
    FROM public.brand_marcas_dataset_treino_v1
"""

MODEL_OUTPUT_DIR = Path("model_output/marcas")
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
    engine = create_engine(db_url, connect_args={'client_encoding': 'utf8'})
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

    # normaliza texto
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
    for marca, cnt in dist.items():
        pct = cnt / len(df) * 100
        bar = "█" * int(pct / 2)
        print(f"   {marca:<25} {cnt:>6,}  {pct:5.1f}%  {bar}")

    return df


# ---------------------------------------------------------------------------
# 3. VETORIZADOR E CLASSIFICADOR
# ---------------------------------------------------------------------------
def build_tfidf() -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 3),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        strip_accents="unicode",
        max_features=5000,
    )

def build_classifier() -> LogisticRegression:
    return LogisticRegression(
        C=1.0,
        max_iter=1000,
        solver="lbfgs",
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


# ---------------------------------------------------------------------------
# 4. TREINAMENTO COM OVERSAMPLING E AVALIAÇÃO
# ---------------------------------------------------------------------------
def train_and_evaluate(df: pd.DataFrame):
    X = df[TEXT_COLUMN]
    y = df[TARGET_COLUMN]

    # encode de labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # distribuição antes do oversampling
    dist_antes = pd.Series(y_enc).value_counts()
    print(f"\n   Classes com menos de 100 exemplos:")
    for idx, cnt in dist_antes[dist_antes < 100].items():
        print(f"   → {le.classes_[idx]:<25} {cnt} exemplos")

    # split estratificado
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc,
        test_size=TEST_SIZE,
        stratify=y_enc,
        random_state=RANDOM_STATE,
    )
    print(f"\n🔀 Split: {len(X_train):,} treino | {len(X_test):,} teste")

    # vetoriza o treino
    print("\n🔢 Vetorizando textos...")
    tfidf = build_tfidf()
    X_train_vec = tfidf.fit_transform(X_train)
    X_test_vec  = tfidf.transform(X_test)

    # OVERSAMPLING: replica classes minoritárias até igualar a mediana
    print("\n⚖️  Aplicando oversampling nas classes minoritárias...")
    mediana = int(np.median(dist_antes.values))
    sampling_strategy = {
        idx: max(cnt, mediana)
        for idx, cnt in dist_antes.items()
        if cnt < mediana
    }
    ros = RandomOverSampler(
        sampling_strategy=sampling_strategy,
        random_state=RANDOM_STATE
    )
    X_train_res, y_train_res = ros.fit_resample(X_train_vec, y_train)

    # mostra efeito nas classes problemáticas
    dist_depois = pd.Series(y_train_res).value_counts()
    for classe in ['sol', 'guarana antarctica', 'original']:
        if classe in le.classes_:
            idx = le.transform([classe])[0]
            antes  = dist_antes.get(idx, 0)
            depois = dist_depois.get(idx, 0)
            print(f"   {classe:<25} {antes:>4} → {depois:>4} exemplos")

    # treina classificador com dados oversampled
    print("\n🚀 Treinando modelo com dados oversampled...")
    clf = build_classifier()
    clf.fit(X_train_res, y_train_res)

    # avalia no teste (sem oversampling — reflete realidade)
    y_pred = clf.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average="macro")

    print(f"\n✅ Resultados no conjunto de teste:")
    print(f"   Acurácia : {acc:.4f}")
    print(f"   F1-macro : {f1:.4f}")
    print(f"\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_, digits=3))

    # monta pipeline final (tfidf já fitado + clf treinado com oversampling)
    pipeline = Pipeline([
        ("tfidf", tfidf),
        ("clf",   clf),
    ])

    # validação cruzada no treino original (estimativa honesta)
    print("\n⏳ Validação cruzada (5 folds)...")
    pipeline_cv = Pipeline([("tfidf", build_tfidf()), ("clf", build_classifier())])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(pipeline_cv, X_train, y_train, cv=cv, scoring="f1_macro", n_jobs=-1)
    print(f"   F1-macro CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    return pipeline, le, acc, f1, cv_scores


# ---------------------------------------------------------------------------
# 5. SALVAMENTO DO MODELO
# ---------------------------------------------------------------------------
def save_artifacts(pipeline: Pipeline, le: LabelEncoder, acc: float, f1: float, cv_scores):
    print("\n💾 Salvando artefatos...")

    joblib.dump(pipeline, MODEL_OUTPUT_DIR / "pipeline.joblib")
    joblib.dump(le,       MODEL_OUTPUT_DIR / "label_encoder.joblib")

    metadata = {
        "classes":           le.classes_.tolist(),
        "n_classes":         len(le.classes_),
        "text_column":       TEXT_COLUMN,
        "target_column":     TARGET_COLUMN,
        "accuracy":          round(acc, 4),
        "f1_macro":          round(f1, 4),
        "cv_f1_macro_mean":  round(cv_scores.mean(), 4),
        "cv_f1_macro_std":   round(cv_scores.std(), 4),
        "model_type":        "TfidfVectorizer (char n-grams) + LogisticRegression + RandomOverSampler",
        "use_case":          "Classificação de marcas de bebidas",
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
    """Recebe uma lista de textos e retorna predições com probabilidades."""
    cleaned = [normalize_text(t) for t in texts]
    proba   = pipeline.predict_proba(cleaned)
    preds   = proba.argmax(axis=1)

    results = []
    for i, (pred, prob_row) in enumerate(zip(preds, proba)):
        top3_idx = np.argsort(prob_row)[::-1][:3]
        top3 = [
            {"marca": le.classes_[j], "confianca": round(float(prob_row[j]), 4)}
            for j in top3_idx
        ]
        results.append({
            "texto_original": texts[i],
            "marca_predita":  le.classes_[pred],
            "confianca":      round(float(prob_row[pred]), 4),
            "top3":           top3,
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