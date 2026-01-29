# API de Classificação e-SIC (Público vs Não Público)

Este projeto disponibiliza uma **API em FastAPI** para **inferência** de um modelo Transformer (BERTimbau fine-tuned) para **classificação binária de pedidos e-SIC**:

- **Público**
- **Não Público**

A API permite:

- Classificação de um único pedido
- Classificação em lote (JSON)
- Classificação via upload de planilha (Excel ou CSV)
- Retorno de contagens agregadas (público vs não público)
- Uso opcional de GPU NVIDIA para acelerar a inferência

> ⚠️ **Este projeto NÃO realiza treinamento do modelo.**  
> Ele utiliza um modelo já treinado e salvo em `model_artifacts/`.

---

## 1) Requisitos

### Sistema

- Windows 10/11, Linux ou macOS
- Acesso ao terminal (PowerShell, CMD ou Bash)

### Python

- **Python 3.11 ou 3.12** (recomendado)

### Hardware (opcional)

- **GPU NVIDIA (opcional)** para inferência acelerada
- CPU funciona normalmente

---

## 2) Estrutura do projeto

```bash
api/
├─ main.py                     # aplicação FastAPI
├─ app/
│  ├─ model_runtime.py         # carregamento do modelo e inferência
│  ├─ schemas.py               # schemas Pydantic
│  └─ __init__.py
├─ model_artifacts/            # modelo treinado (save_pretrained)
├─ requirements.txt            # dependências padrão (CPU)
├─ requirements-gpu.txt        # dependências opcionais para GPU
└─ README.md
```

---

## 3) Criando e ativando a Virtual Environment (venv)

### Windows (PowerShell)

```powershell
cd caminho\para\api
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Se o PowerShell bloquear a ativação:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Depois ative novamente:

```powershell
.\.venv\Scripts\Activate.ps1
```

### Windows (CMD)

```bat
cd caminho\para\api
py -3.12 -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
```

### Linux/macOS

```bash
cd caminho/para/api
python3.12 -m venv .venv
source .venv/Scripts/activate
python -m pip install --upgrade pip
```

> para ativar a venv verifique se está na pasta Scripts ou na pasta bin.

---

## 4) Instalando dependências

### 4.1) Instalação padrão (CPU – recomendado para banca)

```bash
pip install -r requirements.txt
```

### 4.2) Instalação com GPU (opcional)

Com a venv ativada, execute:

```bash
pip install -r requirements-gpu.txt
```

> Observação: se sua GPU/driver exigir uma versão específica de CUDA, use os comandos oficiais do PyTorch.

---

## 5) Verificando GPU (opcional)

```bash
python -c "import torch; print('CUDA disponível:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

---

## 6) Executando a API

```bash
uvicorn main:app --reload
```

A API ficará disponível em:

- **API:** http://127.0.0.1:8000
- **Swagger (documentação interativa):** http://127.0.0.1:8000/docs
- **OpenAPI JSON:** http://127.0.0.1:8000/openapi.json

---

## 7) Rotas disponíveis (com exemplos)

### 7.1) `GET /` (healthcheck)

**Descrição:** Verifica se a API está no ar e informa o device (cpu/cuda).

**Exemplo (curl):**

```bash
curl http://127.0.0.1:8000/
```

**Response (exemplo):**

```json
{
  "status": "ok",
  "service": "e-SIC classifier",
  "device": "cuda"
}
```

---

### 7.2) `GET /info`

**Descrição:** Informações resumidas sobre o modelo e configuração em execução.

**Exemplo (curl):**

```bash
curl http://127.0.0.1:8000/info
```

**Response (exemplo):**

```json
{
  "model": "BERTimbau (fine-tuned)",
  "task": "Binary classification (Public vs Non-Public)",
  "threshold": 0.5,
  "device": "cuda"
}
```

---

### 7.3) `POST /predict`

**Descrição:** Classifica um único pedido.

**Request (JSON):**

```json
{
  "texto": "Solicito informações gerais sobre orçamento."
}
```

**Exemplo (curl):**

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d "{\"texto\":\"Solicito informações gerais sobre orçamento.\"}"
```

**Response (exemplo):**

```json
{
  "label": "publico",
  "score_nao_publico": 0.12,
  "threshold": 0.5
}
```

> Interpretação: `score_nao_publico` é a probabilidade (0–1) da classe **Não Público**.  
> A classificação final depende do `threshold`.

---

### 7.4) `POST /predict-batch`

**Descrição:** Classifica uma lista de textos em lote (JSON). Retorna resultados + contagens.

**Request (JSON):**

```json
{
  "textos": [
    "Pedido de dados estatísticos",
    "Nome completo e CPF do servidor responsável"
  ],
  "threshold": 0.4
}
```

> `threshold` é opcional. Se não for enviado, a API usa o `THRESHOLD` padrão (env/config).

**Exemplo (curl):**

```bash
curl -X POST http://127.0.0.1:8000/predict-batch \
  -H "Content-Type: application/json" \
  -d "{\"textos\":[\"Pedido de dados estatísticos\",\"Nome completo e CPF do servidor responsável\"],\"threshold\":0.4}"
```

**Response (exemplo):**

```json
{
  "resultados": [
    {
      "index": 0,
      "texto": "Pedido de dados estatísticos",
      "label": "publico",
      "score_nao_publico": 0.18
    },
    {
      "index": 1,
      "texto": "Nome completo e CPF do servidor responsável",
      "label": "nao_publico",
      "score_nao_publico": 0.93
    }
  ],
  "qtd_publico": 1,
  "qtd_nao_publico": 1,
  "total": 2,
  "threshold": 0.4
}
```

---

### 7.5) `POST /predict-excel`

**Descrição:** Classifica pedidos enviados via upload de planilha (`.xlsx` ou `.csv`).

**Parâmetros:**

- `file` (obrigatório): arquivo `.xlsx` ou `.csv`
- `coluna_texto` (opcional): nome da coluna onde está o texto.
  - Se informado, a API usa exatamente essa coluna (case-insensitive).
  - Se não informado, a API tenta detectar automaticamente a coluna de texto mais provável.

- `threshold` (opcional): sobrescreve threshold apenas para esta chamada
- `batch_size` (opcional, default=`32`): tamanho do lote na inferência (GPU costuma ser mais rápida com lotes maiores)

#### Exemplo 1: Excel (.xlsx)

```bash
curl -X POST "http://127.0.0.1:8000/predict-excel?coluna_texto=texto&threshold=0.4&batch_size=32" \
  -H "accept: application/json" \
  -F "file=@pedidos.xlsx"
```

#### Exemplo 2: CSV (.csv)

```bash
curl -X POST "http://127.0.0.1:8000/predict-excel?coluna_texto=texto" \
  -H "accept: application/json" \
  -F "file=@pedidos.csv"
```

**Response (exemplo):**

```json
{
  "resultados": [
    {
      "index": 0,
      "texto": "Pedido de dados estatísticos",
      "label": "publico",
      "score_nao_publico": 0.18
    },
    {
      "index": 1,
      "texto": "Nome completo e CPF do servidor responsável",
      "label": "nao_publico",
      "score_nao_publico": 0.93
    }
  ],
  "qtd_publico": 1,
  "qtd_nao_publico": 1,
  "total": 2,
  "threshold": 0.4,
  "coluna_texto": "texto",
  "filename": "pedidos.xlsx"
}
```

---

## 8) Teste pelo Swagger

1. Suba a API (`uvicorn main:app --reload`)
2. Abra: `http://127.0.0.1:8000/docs`
3. Expanda a rota desejada e clique em **Try it out**
4. Para `/predict-excel`, selecione o arquivo no campo **file** e execute

---

## 9) Solução de problemas (Troubleshooting)

### Erro: `ModuleNotFoundError`

- Confirme que a venv está ativada
- Reinstale dependências:

```bash
pip install -r requirements.txt
```

### Erro: upload de arquivo não funciona

Garanta que `python-multipart` está instalado:

```bash
pip install python-multipart
```

### Erro: falha ao ler Excel

Garanta que `openpyxl` está instalado:

```bash
pip install openpyxl
```

### GPU não reconhecida

- A API funciona normalmente em CPU
- Verifique driver NVIDIA e reinstale PyTorch (seção 4.2/5)

---

## 10) Encerrando a venv

```bash
deactivate
```

---

## Comandos rápidos (TL;DR)

### Windows (PowerShell)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn main:app --reload
```

### Linux/macOS

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload
```

## 👥 Equipe e Contexto do Projeto

Este projeto está sendo desenvolvido como parte de uma iniciativa de **inovação e experimentação tecnológica aplicada à gestão pública**, com foco no apoio à **triagem e análise de pedidos de acesso à informação (e-SIC)** por meio de técnicas de **Inteligência Artificial**.

O objetivo é explorar soluções práticas que auxiliem analistas e gestores públicos na tomada de decisão, promovendo maior eficiência, padronização e apoio técnico ao processo.

### Equipe de Desenvolvimento

- **Maikon Santos** — Desenvolvedor Fullstack  
  GitHub: [@Maikon-sant](https://github.com/Maikon-sant)

- **Maysa Santos** — Tech Lead & Desenvolvedora Fullstack  
  GitHub: [@Maysamkt](https://github.com/Maysamkt)
