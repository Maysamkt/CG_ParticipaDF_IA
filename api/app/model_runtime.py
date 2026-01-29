from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


@dataclass(frozen=True)
class InferenceConfig:
    """
    Configuração de inferência do modelo.

    - model_dir: caminho da pasta gerada por save_pretrained() (modelo + tokenizer).
    - max_length: tamanho máximo de tokens por entrada.
    - threshold: limiar para classificar como "nao_publico" (minimiza FN ao reduzir o threshold).
    - device:
        - "auto" -> escolhe "cuda" se disponível, senão "cpu"
        - "cpu"
        - "cuda" / "cuda:0" / "cuda:1" etc.
    """
    model_dir: str
    max_length: int = 256
    threshold: float = 0.50
    device: str = "auto"


class ModelRuntime:
    """
    Runtime de inferência:
    - carrega tokenizer e modelo uma vez
    - expõe métodos para predição individual e em lote
    """

    def __init__(self, cfg: InferenceConfig) -> None:
        # Resolve o device final. Mantemos "auto" no config por conveniência,
        # mas aqui garantimos que o device efetivo será "cpu" ou "cuda(:idx)".
        device = self._resolve_device(cfg.device)

        # Como o dataclass é frozen=True, usamos object.__setattr__ para registrar
        # o device efetivo que será usado durante a execução.
        object.__setattr__(cfg, "device", device)
        self.cfg = cfg

        # Carrega tokenizer/modelo a partir da pasta de artefatos.
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.cfg.model_dir)

        # Envia modelo para o device e coloca em modo eval (desliga dropout etc.)
        self.model.to(self.cfg.device)
        self.model.eval()

    @staticmethod
    def _resolve_device(device: Optional[str]) -> str:
        """
        Resolve a string de device para um valor válido do PyTorch.
        """
        dev = (device or "").strip().lower()

        if dev in ("", "auto", "gpu"):
            return "cuda" if torch.cuda.is_available() else "cpu"

        # Se pediram CUDA mas CUDA não está disponível, cai para CPU.
        if dev.startswith("cuda") and not torch.cuda.is_available():
            return "cpu"

        # Caso geral: "cpu" ou "cuda:0" etc.
        return dev

    @torch.inference_mode()
    def predict_one(self, texto: str) -> Tuple[str, float]:
        """
        Prediz um único texto.

        Retorna:
        - label: "publico" ou "nao_publico"
        - score_nao_publico: probabilidade (0–1) da classe Não Público
        """
        # Reaproveita o caminho de batch para evitar duplicação de lógica.
        label, score = self.predict_many_batch([texto], batch_size=1)[0]
        return label, score

    @torch.inference_mode()
    def predict_many_batch(self, textos: List[str], batch_size: int = 32) -> List[Tuple[str, float]]:
        """
        Prediz uma lista de textos em lotes (batch), mais rápido (especialmente em GPU).

        Retorna uma lista de tuplas (label, score_nao_publico), na mesma ordem de entrada.
        """
        if batch_size <= 0:
            raise ValueError("batch_size deve ser > 0")

        # Normaliza entradas para string e remove espaços.
        # Observação: texto vazio é permitido (quem decide como tratar é a camada da API).
        cleaned = [("" if t is None else str(t).strip()) for t in textos]

        outputs: List[Tuple[str, float]] = []

        for start in range(0, len(cleaned), batch_size):
            chunk = cleaned[start:start + batch_size]

            # Tokenização em lote (padding dinâmico e truncation).
            enc = self.tokenizer(
                chunk,
                truncation=True,
                padding=True,
                max_length=self.cfg.max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(self.cfg.device) for k, v in enc.items()}

            # Forward pass
            logits = self.model(**enc).logits  # shape: [B, 2] (assumindo binário)
            probs = torch.softmax(logits, dim=-1)  # shape: [B, 2]

            # Probabilidade da classe 1: "nao_publico"
            scores_nao_publico = probs[:, 1].detach().cpu().tolist()

            # Converte score em label usando threshold do runtime.
            for score in scores_nao_publico:
                score_f = float(score)
                label = "nao_publico" if score_f >= self.cfg.threshold else "publico"
                outputs.append((label, score_f))

        return outputs
