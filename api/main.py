"""
API de classificação de marcas de bebidas
FastAPI — carrega o modelo uma vez e reutiliza em todas as requisições
"""

import json
import re
import unicodedata
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
MODEL_DIR = Path(__file__).parent / "model_output" / "marcas"

# ---------------------------------------------------------------------------
# CARGA DO MODELO (uma única vez ao subir a API)
# ---------------------------------------------------------------------------
try:
    pipeline = joblib.load(MODEL_DIR / "pipeline.joblib")
    le       = joblib.load(MODEL_DIR / "label_encoder.joblib")
    with open(MODEL_DIR / "metadata.json", encoding="utf-8") as f:
        metadata = json.load(f)
    print(f"✅ Modelo carregado — {metadata['n_classes']} marcas | F1-macro {metadata['f1_macro']}")
except FileNotFoundError:
    raise RuntimeError(
        "Arquivos do modelo não encontrados em model_output/marcas/. "
        "Rode training/train_marcas.py primeiro."
    )

# ---------------------------------------------------------------------------
# APP
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Classificador de Marcas de Bebidas",
    description="Classifica marcas de bebidas a partir do nome normalizado do item",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# SCHEMAS
# ---------------------------------------------------------------------------
class Item(BaseModel):
    texto: str

class ItemBatch(BaseModel):
    itens: list[str]

# ---------------------------------------------------------------------------
# PRÉ-PROCESSAMENTO (igual ao train_marcas.py)
# ---------------------------------------------------------------------------
def normalize_text(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    text = text.lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def predict(texts: list[str]) -> list[dict]:
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
# ROTAS
# ---------------------------------------------------------------------------
@app.get("/")
def root():
    return {
        "status":   "ok",
        "modelo":   metadata["model_type"],
        "marcas":   metadata["classes"],
        "f1_macro": metadata["f1_macro"],
        "acuracia": metadata["accuracy"],
    }

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict_single(item: Item):
    """Classifica um único item."""
    if not item.texto.strip():
        raise HTTPException(status_code=400, detail="Campo 'texto' não pode ser vazio.")
    return predict([item.texto])[0]

@app.post("/predict/batch")
def predict_batch(batch: ItemBatch):
    """Classifica uma lista de itens de uma vez."""
    if not batch.itens:
        raise HTTPException(status_code=400, detail="Lista 'itens' não pode ser vazia.")
    if len(batch.itens) > 50_000:
        raise HTTPException(status_code=400, detail="Máximo de 50.000 itens por requisição.")
    resultados = predict(batch.itens)
    return {
        "total":      len(resultados),
        "resultados": resultados,
    }

@app.get("/marcas")
def listar_marcas():
    """Retorna todas as marcas que o modelo conhece."""
    return {"marcas": metadata["classes"]}