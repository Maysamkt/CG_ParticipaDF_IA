from __future__ import annotations

from typing import List, Literal, Optional
from pydantic import BaseModel, Field

# Labels finais retornados pela API
Label = Literal["publico", "nao_publico"]


# =========================
# Requests
# =========================

class PredictRequest(BaseModel):
    """
    Request para classificação de um único pedido e-SIC.
    """
    texto: str = Field(
        ...,
        min_length=1,
        description="Texto completo do pedido e-SIC a ser classificado",
        examples=["Solicito informações sobre o orçamento anual do órgão."]
    )


class BatchPredictRequest(BaseModel):
    """
    Request para classificação em lote (JSON).
    """
    textos: List[str] = Field(
        ...,
        min_length=1,
        description="Lista de textos a serem classificados",
        examples=[[
            "Pedido de informações gerais sobre despesas",
            "Nome completo e CPF do servidor responsável"
        ]]
    )

    threshold: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description=(
            "Limiar opcional para classificar como 'nao_publico'. "
            "Se informado, sobrescreve o threshold padrão apenas para esta requisição."
        ),
        examples=[0.4]
    )


# =========================
# Responses / DTOs
# =========================

class PredictResponse(BaseModel):
    """
    Response da classificação de um único pedido.
    """
    label: Label = Field(
        ...,
        description="Classificação final do pedido"
    )

    score_nao_publico: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probabilidade (0–1) da classe 'Não Público'"
    )

    threshold: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Threshold utilizado para gerar a classificação"
    )


class BatchItemResult(BaseModel):
    """
    Resultado individual dentro de uma classificação em lote.
    """
    index: int = Field(
        ...,
        ge=0,
        description="Índice do texto na lista de entrada"
    )

    texto: str = Field(
        ...,
        description="Texto original classificado"
    )

    label: Label = Field(
        ...,
        description="Classificação final do texto"
    )

    score_nao_publico: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probabilidade (0–1) da classe 'Não Público'"
    )


class BatchPredictResponse(BaseModel):
    """
    Response da classificação em lote (JSON).
    """
    resultados: List[BatchItemResult] = Field(
        ...,
        description="Lista de resultados individuais"
    )

    qtd_publico: int = Field(
        ...,
        ge=0,
        description="Quantidade de pedidos classificados como Público"
    )

    qtd_nao_publico: int = Field(
        ...,
        ge=0,
        description="Quantidade de pedidos classificados como Não Público"
    )

    total: int = Field(
        ...,
        ge=0,
        description="Total de pedidos processados"
    )

    threshold: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Threshold utilizado na classificação"
    )


class ExcelPredictResponse(BaseModel):
    """
    Response da classificação via upload de Excel/CSV.
    """
    resultados: List[BatchItemResult] = Field(
        ...,
        description="Resultados linha a linha da planilha"
    )

    qtd_publico: int = Field(
        ...,
        ge=0,
        description="Quantidade de linhas classificadas como Público"
    )

    qtd_nao_publico: int = Field(
        ...,
        ge=0,
        description="Quantidade de linhas classificadas como Não Público"
    )

    total: int = Field(
        ...,
        ge=0,
        description="Total de linhas processadas"
    )

    threshold: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Threshold utilizado na classificação"
    )

    coluna_texto: str = Field(
        ...,
        description="Nome da coluna da planilha usada como texto de entrada"
    )

    filename: str = Field(
        ...,
        description="Nome do arquivo enviado"
    )
