from __future__ import annotations

import io
import os
from typing import List, Optional

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware


from app.model_runtime import InferenceConfig, ModelRuntime
from app.schemas import (
    BatchItemResult,
    BatchPredictRequest,
    BatchPredictResponse,
    ExcelPredictResponse,
    PredictRequest,
    PredictResponse,
)


# App


app = FastAPI(title="Classificador e-SIC (Público vs Não-Público)")

# Configuração CORS (Cross-Origin Resource Sharing)
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,       # em dev pode ser ["*"]
    allow_credentials=True,
    allow_methods=["*"],         # libera OPTIONS
    allow_headers=["*"],         # libera preflight headers
)

# Runtime global carregado no startup para evitar recarregar modelo a cada request
runtime: Optional[ModelRuntime] = None



# Startup


@app.on_event("startup")
def load_model() -> None:
    """
    Carrega o tokenizer/modelo uma única vez quando a API inicia.
    """
    global runtime

    model_dir = os.getenv("MODEL_DIR", "model_artifacts")
    threshold = float(os.getenv("THRESHOLD", "0.50"))
    device = os.getenv("DEVICE", "auto")  # "auto", "cpu", "cuda", "cuda:0", ...

    cfg = InferenceConfig(model_dir=model_dir, threshold=threshold, device=device)
    runtime = ModelRuntime(cfg)

    print(f"[INFO] Modelo carregado em: {runtime.cfg.device.upper()}")


def _require_runtime() -> ModelRuntime:
    """
    Garante que o runtime foi carregado (por segurança).
    """
    if runtime is None:
        raise HTTPException(status_code=503, detail="Model runtime not loaded")
    return runtime


# Health / Info


@app.get("/")
def root() -> dict:
    """
    Healthcheck simples.
    """
    rt = runtime
    return {
        "status": "ok",
        "service": "e-SIC classifier",
        "device": rt.cfg.device if rt else "unknown",
    }


@app.get("/info")
def info() -> dict:
    """
    Informações resumidas do serviço/modelo.
    """
    rt = _require_runtime()
    return {
        "model": "BERTimbau (fine-tuned)",
        "task": "Binary classification (Public vs Non-Public)",
        "threshold": rt.cfg.threshold,
        "device": rt.cfg.device,
    }



# Core endpoints


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    """
    Classifica um único pedido.
    """
    rt = _require_runtime()
    label, score = rt.predict_one(req.texto)

    return PredictResponse(
        label=label,
        score_nao_publico=score,
        threshold=rt.cfg.threshold,
    )


@app.post("/predict-batch", response_model=BatchPredictResponse)
def predict_batch(req: BatchPredictRequest) -> BatchPredictResponse:
    """
    Classifica uma lista de textos em lote (JSON) e retorna contagens agregadas.
    """
    rt = _require_runtime()

    # Threshold específico por requisição (opcional)
    thr = req.threshold if req.threshold is not None else rt.cfg.threshold

    # Executa inferência em batch para performance (especialmente em GPU)
    pairs = rt.predict_many_batch(req.textos, batch_size=32)

    resultados: List[BatchItemResult] = []
    qtd_publico = 0
    qtd_nao_publico = 0

    for i, texto in enumerate(req.textos):
        texto_limpo = (texto or "").strip()
        if not texto_limpo:
            # Aqui preferimos falhar rápido para indicar erro no payload.
            raise HTTPException(status_code=400, detail=f"Texto vazio no índice {i}")

        # Pegamos apenas o score e recalculamos label com o threshold da request
        _, score = pairs[i]
        label_final = "nao_publico" if score >= thr else "publico"

        if label_final == "publico":
            qtd_publico += 1
        else:
            qtd_nao_publico += 1

        resultados.append(
            BatchItemResult(
                index=i,
                texto=texto_limpo,
                label=label_final,
                score_nao_publico=score,
            )
        )

    return BatchPredictResponse(
        resultados=resultados,
        qtd_publico=qtd_publico,
        qtd_nao_publico=qtd_nao_publico,
        total=len(resultados),
        threshold=thr,
    )


@app.post("/predict-excel", response_model=ExcelPredictResponse)
async def predict_excel(
    file: UploadFile = File(...),
    coluna_texto: Optional[str] = None,
    threshold: Optional[float] = None,
    batch_size: int = 32,
) -> ExcelPredictResponse:
    """
    Classifica pedidos enviados via upload de planilha (.xlsx ou .csv).

    - coluna_texto: nome da coluna com o texto do pedido
    - threshold: opcional (sobrescreve apenas esta chamada)
    - batch_size: controla o tamanho dos lotes na inferência
    """
    rt = _require_runtime()

    if batch_size <= 0:
        raise HTTPException(status_code=400, detail="batch_size deve ser > 0")

    filename = file.filename or "arquivo"
    content = await file.read()

    # Threshold específico por requisição (opcional)
    thr = threshold if threshold is not None else rt.cfg.threshold

    # Leitura do arquivo
    try:
        lower = filename.lower()
        if lower.endswith(".xlsx"):
            df = pd.read_excel(io.BytesIO(content))
        elif lower.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(content))
        else:
            raise HTTPException(status_code=400, detail="Envie um arquivo .xlsx ou .csv")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Falha ao ler arquivo: {e}")

    
    
    # Normaliza nomes de coluna (tira espaços invisíveis)
    df.columns = [str(c).strip() for c in df.columns]

    def _norm(s: str) -> str:
        return "".join(ch for ch in s.strip().lower() if ch.isalnum())

    def _pick_text_column(df: pd.DataFrame) -> str:
        cols = list(df.columns)

        # Preferências por nome (inclui seu caso real)
        preferred = ["Texto Mascarado", "texto", "mensagem", "descricao", "solicitacao", "pedido"]
        cols_norm_map = {_norm(str(c)): str(c) for c in cols}

        for p in preferred:
            key = _norm(p)
            if key in cols_norm_map:
                return cols_norm_map[key]

        # Fallback: coluna com maior "densidade" de texto
        best_col = None
        best_score = -1.0
        for c in cols:
            s = df[c].astype(str).fillna("")
            non_empty = (s.str.strip() != "").mean()
            avg_len = s.str.len().mean()
            score = float(non_empty) * float(avg_len)
            if score > best_score:
                best_score = score
                best_col = str(c)

        if not best_col:
            raise HTTPException(status_code=400, detail="Não foi possível detectar a coluna de texto.")
        return best_col

    # Decide qual coluna usar
    col_raw = (coluna_texto or "").strip()

    if not col_raw:
        # Se vier vazio/None, tenta detectar automaticamente
        coluna_texto_final = _pick_text_column(df)
    else:
        # Resolve por comparação "normalizada" (case-insensitive e ignora espaços/sinais)
        cols = list(df.columns)
        norm_map = {_norm(str(c)): str(c) for c in cols}

        key = _norm(col_raw)
        if key in norm_map:
            coluna_texto_final = norm_map[key]
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Coluna '{col_raw}' não encontrada. Colunas disponíveis: {list(df.columns)}",
            )

    textos = df[coluna_texto_final].astype(str).fillna("").tolist()

    pairs = rt.predict_many_batch(textos, batch_size=batch_size)

    resultados: List[BatchItemResult] = []
    qtd_publico = 0
    qtd_nao_publico = 0

    for i, texto in enumerate(textos):
        texto_limpo = (texto or "").strip()

        if not texto_limpo:
            resultados.append(BatchItemResult(index=i, texto="", label="publico", score_nao_publico=0.0))
            qtd_publico += 1
            continue

        _, score = pairs[i]
        label_final = "nao_publico" if score >= thr else "publico"

        if label_final == "publico":
            qtd_publico += 1
        else:
            qtd_nao_publico += 1

        resultados.append(
            BatchItemResult(index=i, texto=texto_limpo, label=label_final, score_nao_publico=score)
        )

    return ExcelPredictResponse(
        resultados=resultados,
        qtd_publico=qtd_publico,
        qtd_nao_publico=qtd_nao_publico,
        total=len(resultados),
        threshold=thr,
        coluna_texto=coluna_texto_final,
        filename=filename,
    )