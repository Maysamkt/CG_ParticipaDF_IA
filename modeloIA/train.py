"""
Treinamento de classificador BINÁRIO para pedidos e-SIC (Público vs Não-Público)

O que este script faz:
1) Carrega uma planilha Excel com textos reais e rótulos (multiclasse) e converte para binário:
   - 0 -> Público
   - 1/2/... -> Não-Público
2) Separa os dados reais em train/val/test (test é holdout REAL para avaliação final).
3) Gera dados sintéticos:
   - Público (sem PII)
   - Não-Público (com PII)
   E valida com detector de PII baseado em regras (regex + palavras-chave).
4) Treina BERTimbau (neuralmind/bert-base-portuguese-cased) com loss ponderada (para desbalanceamento).
5) Aplica early stopping e salva o melhor modelo/tokenizer.
6) Avalia no holdout real e faz varredura de limiar (threshold) para reduzir FN (vazamento).

Requisitos principais:
- pandas, numpy, scikit-learn, torch, transformers, faker, matplotlib

Observação:
- O arquivo Excel precisa ter uma coluna de texto (ex.: "Texto Mascarado", "Pedido", etc.)
  e uma coluna de rótulo (ex.: "LABEL", "Classificação", etc.). O script tenta detectar automaticamente.
"""

from __future__ import annotations

import argparse
import random
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from faker import Faker
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (
    BertForSequenceClassification,
    BertTokenizerFast,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

# CONFIGURAÇÃO


@dataclass(frozen=True)
class Config:
    """Configurações gerais do treinamento."""
    model_name: str = "neuralmind/bert-base-portuguese-cased"
    random_seed: int = 42

    # Quantidade de amostras sintéticas por classe
    n_syn_public: int = 800
    n_syn_nonpublic: int = 800

    # Pastas/arquivos
    output_dir: str = "runs/best_model"

    # Treinamento
    max_length: int = 128
    learning_rate: float = 2e-5
    train_batch_size: int = 16
    eval_batch_size: int = 16
    num_train_epochs: int = 10
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    logging_steps: int = 50

    # Early stopping
    early_stopping_patience: int = 3

    # Split real
    test_size: float = 0.2
    val_size_within_train: float = 0.2


def set_all_seeds(seed: int) -> None:
    """Garante reprodutibilidade (o máximo possível) definindo seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Se estiver em GPU e quiser determinismo extra:
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



# DETECTOR DE PII (REGEX + PALAVRAS-CHAVE)


class PIIDetector:
    """
    Detector simples (rule-based) para PII comum no Brasil.
    Útil para:
    - Garantir que o sintético "público" não venha contaminado com PII
    - Validar se o sintético "não-público" tem PII detectável
    """

    def __init__(self) -> None:
        self.patterns: Dict[str, re.Pattern] = {
            "cpf": re.compile(r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b"),
            "cnpj": re.compile(r"\b\d{2}\.?\d{3}\.?\d{3}/?\d{4}-?\d{2}\b"),
            "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
            "phone": re.compile(r"\b\(?\d{2}\)?\s?\d{4,5}-?\d{4}\b"),
            "cep": re.compile(r"\b\d{5}-?\d{3}\b"),
            "rg": re.compile(r"\bRG:?\s?\d{1,2}\.?\d{3}\.?\d{3}-?[0-9X]\b", re.IGNORECASE),
        }

        self.keywords: List[str] = [
            "cpf", "cnpj", "rg", "carteira", "identidade", "telefone",
            "celular", "email", "endereço", "endereco", "cep", "nascimento",
            "matrícula", "matricula", "senha", "password", "conta bancária",
            "conta bancaria", "agência", "agencia", "pis", "pasep",
        ]

    def contains_pii(self, text: object) -> bool:
        """Retorna True se o texto parece conter PII."""
        if pd.isna(text):
            return False

        s = str(text).lower()

        # Checagem por regex
        for pattern in self.patterns.values():
            if pattern.search(s):
                return True

        # Checagem por palavras-chave
        return any(k in s for k in self.keywords)



# LEITURA DO EXCEL + DETECÇÃO DE COLUNAS + BINARIZAÇÃO


def detect_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    """
    Tenta descobrir automaticamente:
    - a coluna de texto
    - a coluna de rótulo

    Ele tenta match exato (case-insensitive) e depois “contém”.
    """
    possible_text = ["Texto Mascarado", "Pedido", "Texto", "Mensagem", "Conteúdo", "Conteudo"]
    possible_label = ["LABEL", "Classificação", "Classificacao", "Classe", "Risco"]

    def find_col(candidates: Sequence[str]) -> Optional[str]:
        # Match exato
        for cand in candidates:
            for col in df.columns:
                if cand.strip().lower() == str(col).strip().lower():
                    return col
        # Fallback: “contém”
        for cand in candidates:
            for col in df.columns:
                if cand.strip().lower() in str(col).strip().lower():
                    return col
        return None

    return find_col(possible_text), find_col(possible_label)


def load_real_data(
    excel_path: str,
    text_column_override: Optional[str] = None,
    label_column_override: Optional[str] = None,
) -> pd.DataFrame:
    """
    Carrega dados reais do Excel e retorna um DataFrame padronizado com:
    - Pedido (texto)
    - binary_label (0/1)
    """
    df = pd.read_excel(excel_path)
    print(f"\n✓ Carregadas {len(df)} linhas do Excel: {excel_path}")
    print("  Colunas encontradas:", list(df.columns))

    # Descobrir colunas (ou usar override)
    if text_column_override and label_column_override:
        text_col, label_col = text_column_override, label_column_override
    else:
        text_col, label_col = detect_columns(df)

    if text_col is None:
        raise ValueError("Não detectei a coluna de texto. Use --text-col para definir manualmente.")
    if label_col is None:
        raise ValueError("Não detectei a coluna de rótulo. Use --label-col para definir manualmente.")

    print(f"✓ Coluna de texto detectada: {text_col}")
    print(f"✓ Coluna de rótulo detectada: {label_col}")

    # Remover linhas sem texto/rótulo
    df = df.dropna(subset=[text_col, label_col]).copy()

    # Converter para int com segurança
    df[label_col] = df[label_col].astype(int)

    # Distribuição original (multiclasse)
    print("  Distribuição original:", df[label_col].value_counts().to_dict())

    # BINARIZAÇÃO:
    # 0 -> público
    # !=0 -> não-público
    df["binary_label"] = df[label_col].apply(lambda x: 0 if x == 0 else 1)

    print("  Distribuição binária:", df["binary_label"].value_counts().to_dict())

    # Padronizar nome da coluna de texto
    df = df.rename(columns={text_col: "Pedido"})
    return df[["Pedido", "binary_label"]]


def split_real_data(
    df_real: pd.DataFrame,
    cfg: Config,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Faz split apenas nos dados reais:
    - train_real
    - val_real
    - test_real (holdout final)
    """
    train_df, test_df = train_test_split(
        df_real,
        test_size=cfg.test_size,
        random_state=cfg.random_seed,
        stratify=df_real["binary_label"],
    )

    train_df, val_df = train_test_split(
        train_df,
        test_size=cfg.val_size_within_train,
        random_state=cfg.random_seed,
        stratify=train_df["binary_label"],
    )

    print("\n✓ Split REAL (train/val/test):")
    print("  Train:", len(train_df), train_df["binary_label"].value_counts().to_dict())
    print("  Val  :", len(val_df), val_df["binary_label"].value_counts().to_dict())
    print("  Test :", len(test_df), test_df["binary_label"].value_counts().to_dict())
    return train_df, val_df, test_df



# GERAÇÃO DE DADOS SINTÉTICOS


def generate_nonpublic_text(fake: Faker) -> str:
    """Gera textos NÃO-PÚBLICOS (com PII)."""
    name = fake.name()
    cpf = fake.cpf()
    phone = fake.phone_number()
    email = fake.email()
    address = fake.address().replace("\n", ", ")

    def generate_rg() -> str:
        base = f"{random.randint(10, 99)}.{random.randint(100, 999)}.{random.randint(100, 999)}"
        dv = random.randint(0, 9)
        return f"{base}-{dv}"

    rg = generate_rg()

    templates = [
        f"Dados pessoais de {name}, CPF {cpf}, telefone {phone}.",
        f"Informações sobre o servidor {name}, matrícula {fake.random_number(digits=6)}.",
        f"Contato: {name}, e-mail {email}, telefone {phone}.",
        f"Processo individual de {name}, CPF {cpf}, endereço {address}.",
        f"Prontuário médico de {name}, nascimento {fake.date_of_birth()}.",
        f"Dados bancários de {name}: agência {fake.random_number(digits=4)}, conta {fake.random_number(digits=8)}.",
        f"Certidão de {name}, RG {rg}, CPF {cpf}.",
        f"Senha de acesso para {name}: {fake.password()}.",
    ]
    return str(np.random.choice(templates))


def generate_public_text() -> str:
    """
    Gera textos PÚBLICOS (sem PII).
    Importante:
    - Evitar nomes próprios, e-mails, CPF, etc.
    - Trabalhar com pedidos agregados, listas e relatórios, processos sem identificação de pessoa.
    """
    topics = [
        "licitações", "contratos", "orçamento", "execução orçamentária",
        "programas sociais", "fila de atendimento", "obras públicas",
        "relatórios de auditoria", "dados estatísticos", "transparência",
    ]

    orgs = [
        "Secretaria de Saúde", "Secretaria de Educação", "Secretaria de Obras",
        "Administração Regional", "Controladoria-Geral", "Departamento de Trânsito",
    ]

    years = ["2021", "2022", "2023", "2024", "2025"]
    values = [10000, 25000, 50000, 120000, 300000, 1000000]

    year = str(np.random.choice(years))

    # "Número de processo" fictício, sem identificar pessoa
    sei = (
        f"{np.random.randint(10000, 99999)}-"
        f"{np.random.randint(10000000, 99999999)}/{year}-"
        f"{np.random.randint(10, 99)}"
    )

    templates = [
        "Solicito informações sobre {topic} no âmbito da {org} referente ao ano {year}.",
        "Gostaria de obter dados agregados sobre {topic} no Distrito Federal no período de {year}.",
        "Pedido de cópia do relatório de {topic} da {org} referente ao ano {year}.",
        "Solicito estatísticas consolidadas sobre {topic} em {year} (sem identificação de pessoas).",
        "Solicito informações sobre a licitação relacionada a {topic} com valor estimado de R$ {value}.",
        "Peço informações sobre o andamento do processo SEI nº {sei}.",
        "Solicito a lista de contratos firmados pela {org} relacionados a {topic} no ano {year}.",
        "Gostaria de saber quais ações de {topic} foram executadas pela {org} no ano {year}.",
    ]

    return str(np.random.choice(templates)).format(
        topic=np.random.choice(topics),
        org=np.random.choice(orgs),
        year=year,
        value=f"{np.random.choice(values):,}".replace(",", "."),
        sei=sei,
    )


def generate_synthetic_data(cfg: Config, pii_detector: PIIDetector) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Gera dois DataFrames sintéticos:
    - public_df: rótulo 0
    - nonpublic_df: rótulo 1

    E garante que o sintético público NÃO contenha PII (com regeneração).
    """
    fake = Faker("pt_BR")
    Faker.seed(cfg.random_seed)

    print(f"\n✓ Gerando {cfg.n_syn_public} amostras sintéticas PÚBLICAS...")
    public_df = pd.DataFrame({
        "Pedido": [generate_public_text() for _ in range(cfg.n_syn_public)],
        "binary_label": 0,
    })

    print(f"✓ Gerando {cfg.n_syn_nonpublic} amostras sintéticas NÃO-PÚBLICAS...")
    nonpublic_df = pd.DataFrame({
        "Pedido": [generate_nonpublic_text(fake) for _ in range(cfg.n_syn_nonpublic)],
        "binary_label": 1,
    })

    # Filtra público contaminado com PII e repõe até bater a meta
    mask_pii_pub = public_df["Pedido"].apply(pii_detector.contains_pii)
    bad = int(mask_pii_pub.sum())

    if bad > 0:
        print(f"⚠️ Removendo {bad} amostras PÚBLICAS sintéticas com PII (contaminariam o treino).")
        public_df = public_df.loc[~mask_pii_pub].reset_index(drop=True)

    max_tries = 20000
    tries = 0
    while len(public_df) < cfg.n_syn_public:
        tries += 1
        if tries > max_tries:
            raise RuntimeError(
                f"Não consegui regenerar public sem PII após {max_tries} tentativas. "
                "Revise generate_public_text() e/ou PIIDetector."
            )
        t = generate_public_text()
        if not pii_detector.contains_pii(t):
            public_df = pd.concat([public_df, pd.DataFrame([{"Pedido": t, "binary_label": 0}])], ignore_index=True)

    # Relatórios de sanity check
    pii_in_public = int(public_df["Pedido"].apply(pii_detector.contains_pii).sum())
    pii_in_nonpublic = int(nonpublic_df["Pedido"].apply(pii_detector.contains_pii).sum())

    print(f"✓ Checagem PII em público sintético: {pii_in_public}/{len(public_df)}")
    print(f"✓ Checagem PII em não-público sintético: {pii_in_nonpublic}/{len(nonpublic_df)} (esperado alto)")

    return public_df, nonpublic_df



# DATASET PARA TRANSFORMERS


class ESICDataset(Dataset):
    """Dataset PyTorch simples para textos tokenizados e labels."""

    def __init__(self, texts: Sequence[str], labels: Sequence[int], tokenizer: BertTokenizerFast, max_length: int):
        self.encodings = tokenizer(
            list(texts),
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

    def __len__(self) -> int:
        return len(self.labels)



# TREINADOR COM LOSS PONDERADA (DESBALANCEAMENTO)


class WeightedTrainer(Trainer):
    """
    Trainer customizado para aplicar CrossEntropyLoss com pesos por classe.
    Isso ajuda quando a classe "1" (não-público) é muito menos frequente (ou vice-versa).
    """

    def __init__(self, *args, class_weights: torch.Tensor, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        inputs = {k: v for k, v in inputs.items() if k != "labels"}

        outputs = model(**inputs)
        logits = outputs.logits

        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    """
    Métricas binárias padrão:
    - accuracy, precision, recall, f1
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, average="binary", zero_division=0),
        "recall": recall_score(labels, preds, average="binary", zero_division=0),
        "f1": f1_score(labels, preds, average="binary", zero_division=0),
    }


# AVALIAÇÃO COM THRESHOLD (FOCO EM EVITAR FN = VAZAMENTO)


def sweep_thresholds(
    p_nonpublic: np.ndarray,
    y_true: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
) -> Tuple[List[Tuple[float, int, int, int, int, float, float, float]], Optional[float]]:
    """
    Varre thresholds e retorna:
    - rows: lista de tuplas (thr, tn, fp, fn, tp, precision, recall, f1)
    - best_thr: melhor limiar sob a regra:
        1) Preferir FN=0 (zero vazamento)
        2) Entre esses, minimizar FP
        3) Empate -> maior F1
    """
    if thresholds is None:
        thresholds = np.round(np.linspace(0.05, 0.95, 19), 2)

    rows = []
    best_thr = None
    best_key = None  # (fp, -f1)

    for thr in thresholds:
        y_pred = (p_nonpublic >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1v = f1_score(y_true, y_pred, zero_division=0)

        rows.append((float(thr), int(tn), int(fp), int(fn), int(tp), float(prec), float(rec), float(f1v)))

        if fn == 0:
            key = (fp, -f1v)
            if best_key is None or key < best_key:
                best_key = key
                best_thr = float(thr)

    return rows, best_thr


def run_final_evaluation_with_threshold(trainer: Trainer, test_dataset: Dataset, test_labels: np.ndarray) -> None:
    """
    Faz:
    - evaluate padrão (argmax)
    - predict para obter probabilidades
    - sweep de threshold
    - relatório final (com threshold escolhido)
    """
    print("\n" + "=" * 80)
    print("AVALIAÇÃO FINAL NO TESTE REAL (HOLDOUT)")
    print("=" * 80 + "\n")

    # Avaliação padrão do Trainer (usa argmax internamente)
    _ = trainer.evaluate(eval_dataset=test_dataset)

    # Predições (logits -> probabilidades)
    pred_out = trainer.predict(test_dataset)
    logits = pred_out.predictions  # (N, 2)

    probs = torch.softmax(torch.tensor(logits), dim=-1).cpu().numpy()
    p_nonpublic = probs[:, 1]

    # Sweep thresholds
    rows, best_thr = sweep_thresholds(p_nonpublic, test_labels)

    print("THRESHOLD SWEEP (resumo):")
    for thr, tn, fp, fn, tp, prec, rec, f1v in rows:
        print(f"thr={thr:.2f} | FN={fn} FP={fp} | P={prec:.3f} R={rec:.3f} F1={f1v:.3f}")

    # Escolha final do threshold
    if best_thr is not None:
        thresh = best_thr
        print(f"\n✅ Melhor threshold com FN=0 encontrado: {thresh:.2f} (minimiza FP entre os testados)")
    else:
        thresh = 0.35
        print(f"\n⚠️ Nenhum threshold testado zerou FN. Usando fallback THRESH={thresh:.2f}")

    # Predição final com threshold escolhido
    y_pred = (p_nonpublic >= thresh).astype(int)

    # Métricas finais
    acc = accuracy_score(test_labels, y_pred)
    prec = precision_score(test_labels, y_pred, zero_division=0)
    rec = recall_score(test_labels, y_pred, zero_division=0)
    f1v = f1_score(test_labels, y_pred, zero_division=0)
    cm = confusion_matrix(test_labels, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print("\nMÉTRICAS FINAIS (com threshold):")
    print(f"  Threshold: {thresh:.2f}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1-Score : {f1v:.4f}")
    print(f"  FN (vazamento, Non-Public predito como Public): {fn}")
    print(f"  FP (Public predito como Non-Public): {fp}")

    print("\nMATRIZ DE CONFUSÃO:")
    print("                Predito")
    print("                Public  Non-Public")
    print(f"Real Public       {tn:4d}    {fp:4d}")
    print(f"Real Non-Public   {fn:4d}    {tp:4d}")

    print("\nRELATÓRIO DETALHADO:")
    print(classification_report(
        test_labels,
        y_pred,
        target_names=["Public", "Non-Public"],
        digits=4,
        zero_division=0,
    ))



# MAIN: ORQUESTRAÇÃO DO PIPELINE


def parse_args() -> argparse.Namespace:
    """Lê argumentos de linha de comando para tornar o script reutilizável."""
    parser = argparse.ArgumentParser(description="Treino binário BERTimbau para e-SIC.")
    parser.add_argument("--data", type=str, required=True, help="Caminho do Excel (ex.: AMOSTRA_e-SIC.xlsx).")
    parser.add_argument("--output", type=str, default=Config().output_dir, help="Pasta de saída do melhor modelo.")
    parser.add_argument("--text-col", type=str, default=None, help="Override da coluna de texto.")
    parser.add_argument("--label-col", type=str, default=None, help="Override da coluna de rótulo.")
    parser.add_argument("--syn-public", type=int, default=Config().n_syn_public, help="Qtd sintética público.")
    parser.add_argument("--syn-nonpublic", type=int, default=Config().n_syn_nonpublic, help="Qtd sintética não-público.")
    return parser.parse_args()


def main() -> None:
    cfg_base = Config()
    args = parse_args()

    # Atualiza config com argumentos
    cfg = Config(
        model_name=cfg_base.model_name,
        random_seed=cfg_base.random_seed,
        n_syn_public=args.syn_public,
        n_syn_nonpublic=args.syn_nonpublic,
        output_dir=args.output,
        max_length=cfg_base.max_length,
        learning_rate=cfg_base.learning_rate,
        train_batch_size=cfg_base.train_batch_size,
        eval_batch_size=cfg_base.eval_batch_size,
        num_train_epochs=cfg_base.num_train_epochs,
        weight_decay=cfg_base.weight_decay,
        warmup_ratio=cfg_base.warmup_ratio,
        logging_steps=cfg_base.logging_steps,
        early_stopping_patience=cfg_base.early_stopping_patience,
        test_size=cfg_base.test_size,
        val_size_within_train=cfg_base.val_size_within_train,
    )

    print("=" * 80)
    print("TREINAMENTO BINÁRIO (e-SIC) | Público (0) vs Não-Público (1)")
    print("=" * 80)

    set_all_seeds(cfg.random_seed)

    # Inicializa detector de PII
    pii_detector = PIIDetector()
    print("\n✓ Detector de PII inicializado (regex + palavras-chave)")

    # Carrega e prepara dados reais
    real_df = load_real_data(args.data, args.text_col, args.label_col)
    real_train_df, real_val_df, real_test_df = split_real_data(real_df, cfg)

    # Gera sintéticos
    syn_public_df, syn_nonpublic_df = generate_synthetic_data(cfg, pii_detector)

    # Monta treino final (real_train + sintéticos)
    combined_train_df = pd.concat([
        real_train_df.assign(source="real"),
        syn_public_df.assign(source="synthetic_public"),
        syn_nonpublic_df.assign(source="synthetic_nonpublic"),
    ], ignore_index=True)

    print(f"\n✓ Dataset de TREINO combinado: {len(combined_train_df)} amostras")
    print("  Distribuição no treino:", combined_train_df["binary_label"].value_counts().to_dict())

    train_texts = combined_train_df["Pedido"].astype(str).values
    train_labels = combined_train_df["binary_label"].values.astype(int)

    val_texts = real_val_df["Pedido"].astype(str).values
    val_labels = real_val_df["binary_label"].values.astype(int)

    test_texts = real_test_df["Pedido"].astype(str).values
    test_labels = real_test_df["binary_label"].values.astype(int)

    # Tokenizer + Datasets
    tokenizer = BertTokenizerFast.from_pretrained(cfg.model_name)
    print(f"\n✓ Tokenizer carregado: {cfg.model_name}")

    train_dataset = ESICDataset(train_texts, train_labels, tokenizer, cfg.max_length)
    val_dataset = ESICDataset(val_texts, val_labels, tokenizer, cfg.max_length)
    test_dataset = ESICDataset(test_texts, test_labels, tokenizer, cfg.max_length)
    print("✓ Datasets PyTorch criados (train/val/test)")

    # Pesos de classe (para loss ponderada)
    class_counts = np.bincount(train_labels, minlength=2)
    total = int(class_counts.sum())

    # Fórmula: total / (2 * count_classe)
    # Intuição: quanto menor a classe, maior o peso.
    class_weights = torch.tensor(
        [total / (2 * c) if c > 0 else 0.0 for c in class_counts],
        dtype=torch.float32,
    )
    print(f"\n✓ Pesos de classe: {class_weights.tolist()} (counts={class_counts.tolist()})")

    # Modelo
    model = BertForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=2,
        problem_type="single_label_classification",
        use_safetensors=True,
    )
    print("✓ Modelo BERTimbau carregado (2 classes)")

    # TrainingArguments
    # Nota de compatibilidade:
    # Em versões mais antigas pode ser "evaluation_strategy". Aqui usamos "eval_strategy"
    # como no seu script original; se der erro, troque para evaluation_strategy="epoch".
    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.train_batch_size,
        per_device_eval_batch_size=cfg.eval_batch_size,
        num_train_epochs=cfg.num_train_epochs,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        logging_steps=cfg.logging_steps,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        seed=cfg.random_seed,
        report_to="none",
    )

    print("\n✓ Configuração de treino:")
    print(f"  Epochs       : {cfg.num_train_epochs}")
    print(f"  LR           : {cfg.learning_rate}")
    print(f"  Batch (train): {cfg.train_batch_size}")
    print(f"  Batch (eval) : {cfg.eval_batch_size}")
    print(f"  Output dir   : {cfg.output_dir}")

    # Trainer com early stopping
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg.early_stopping_patience)],
        class_weights=class_weights,
    )

    print("\n" + "=" * 80)
    print("INICIANDO TREINAMENTO")
    print("=" * 80 + "\n")

    train_result = trainer.train()

    print("\n" + "=" * 80)
    print("TREINAMENTO FINALIZADO")
    print("=" * 80)
    print(f"Tempo (s): {train_result.metrics.get('train_runtime', 0.0):.2f}")
    print(f"Loss     : {train_result.metrics.get('train_loss', 0.0):.4f}")

    # Salva melhor modelo + tokenizer
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    print(f"\n✓ Melhor modelo e tokenizer salvos em: {cfg.output_dir}")

    # Avaliação final no holdout real com threshold
    run_final_evaluation_with_threshold(trainer, test_dataset, test_labels)

    print("\n" + "=" * 80)
    print("SCRIPT CONCLUÍDO COM SUCESSO")
    print("=" * 80)
    print(f"Modelo final em: {cfg.output_dir}")


if __name__ == "__main__":
    main()
