# Treinamento do Modelo (Classificação Binária e-SIC)

Este projeto treina um modelo Transformer (BERTimbau) para **classificação binária**:

- **0 = Público**
- **1 = Não-Público**

O script lê um arquivo Excel, faz split (train/val/test), gera dados sintéticos (público e não-público), treina com `transformers` e salva o melhor modelo.

---

## 1) Requisitos

### Sistema

- Windows 10/11, Linux ou macOS
- Acesso ao terminal (PowerShell, CMD, Bash)

### Python

- **Python 3.11 ou 3.12** (recomendado)
  - Observação: dependendo do sistema e das versões de CUDA/PyTorch, podem ocorrer incompatibilidades.

### Hardware (opcional)

- **GPU NVIDIA (recomendado)** para acelerar bastante o treino
- Treino em CPU funciona, mas é mais lento

---

## 2) Estrutura sugerida do projeto

```bash
seu-projeto/
├─ train.py                     # script de treino
├─ requirements.txt             # dependências
├─ data/
│  └─ AMOSTRA_e-SIC.xlsx        # seu arquivo Excel
└─ runs/
   └─ best_model/               # saída padrão do modelo
```

> Se o seu script tiver outro nome (ex.: `train_clean.py`), substitua `train.py` pelos comandos abaixo.

---

## 3) Criando e ativando a Virtual Environment (venv)

Boas práticas: sempre usar venv para isolar dependências do projeto.

### Windows (PowerShell)

```powershell
cd caminho\para\seu-projeto
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Se o PowerShell bloquear a ativação, rode:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Depois tente ativar novamente:

```powershell
.\.venv\Scripts\Activate.ps1
```

### Windows (CMD)

```bat
cd caminho\para\seu-projeto
py -3.12 -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
```

### Linux/macOS

```bash
cd caminho/para/seu-projeto
python3.12 -m venv .venv
source .venv/Scripts/activate
python -m pip install --upgrade pip
```

> Pode ser que a ativação esteja na pasta bin, nesse caso substitua Scripts por bin.

## 4) Instalando dependências

### 4.1) Usando `requirements.txt` (recomendado)

Instale as dependências:

```bash
pip install -r requirements.txt
```

> Exemplo de `requirements.txt` (mínimo recomendado):

```bash
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
torch
transformers>=4.41
accelerate>=0.31
faker>=24.0
openpyxl>=3.1
```

---

### 4.2) Instalação com GPU (opcional)

Com a venv ativada, execute:

```bash
pip install -r requirements-gpu.txt
```

> O arquivo `requirements-gpu.txt` contém versões de PyTorch com suporte a CUDA, conforme a plataforma.

---

## 5) Configuração de GPU (NVIDIA) (opcional, recomendado)

### 5.1) Como verificar se o PyTorch está vendo a GPU

Com a venv ativada:

```bash
python -c "import torch; print('CUDA disponível:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

- Se aparecer **CUDA disponível: True**, você já está pronto.
- Se aparecer **False**, veja a seção 5.2.

### 5.2) Se `torch.cuda.is_available()` der False

Isso normalmente significa uma destas situações:

- Driver NVIDIA não instalado/atualizado
- CUDA não compatível com a versão do PyTorch instalada
- Você instalou PyTorch CPU-only

Recomendação prática (boas práticas):

- Instale/atualize o driver NVIDIA.
- Reinstale o PyTorch com suporte a CUDA conforme a documentação oficial do PyTorch (variável conforme sistema/CUDA).

**Importante:** não é necessário “ativar GPU no código”. O `transformers` usa GPU automaticamente quando o PyTorch detecta CUDA.

---

## 6) Preparando os dados

Coloque seu Excel em `data/`, por exemplo:

```bash
data/AMOSTRA_e-SIC.xlsx
```

O Excel deve conter:

- Uma coluna com o texto do pedido (ex.: `Texto Mascarado`, `Pedido`, `Texto`, etc.)
- Uma coluna com o rótulo original (ex.: `LABEL`, `Classificação`, etc.)

O script tenta detectar automaticamente essas colunas. Se não encontrar, você pode informar manualmente via CLI.

---

## 7) Rodando o treinamento

### 7.1) Execução padrão

```bash
python train.py --data "data/AMOSTRA_e-SIC.xlsx" --output "runs/best_model"
```

### 7.2) Definindo colunas manualmente (se necessário)

#### Windows (PowerShell)

Opção A (recomendada, em **uma linha**):

```powershell
python train.py --data "data/AMOSTRA_e-SIC.xlsx" --text-col "Texto Mascarado" --label-col "LABEL" --output "runs/best_model"
```

Opção B (com quebra de linha no PowerShell)

> Atenção: a crase ( ` ) **não pode ter espaço depois dela**.

```powershell
python train.py `
  --data "data/AMOSTRA_e-SIC.xlsx" `
  --text-col "Texto Mascarado" `
  --label-col "LABEL" `
  --output "runs/best_model"
```

#### Linux/macOS

```bash
python train.py \
  --data "data/AMOSTRA_e-SIC.xlsx" \
  --text-col "Texto Mascarado" \
  --label-col "LABEL" \
  --output "runs/best_model"
```

### 7.3) Ajustando quantidade de dados sintéticos

```bash
python train.py --data "data/AMOSTRA_e-SIC.xlsx" --syn-public 800 --syn-nonpublic 800
```

---

## 8) Saídas do treinamento

Ao final, o script salva:

- Modelo (pesos) e tokenizer em:
  - `runs/best_model/` (ou no diretório que você passar em `--output`)

E imprime no terminal:

- Distribuição de classes
- Métricas de validação por época
- Avaliação final no teste real (holdout)
- Varredura de threshold (para reduzir FN / vazamento)

---

## 9) Solução de problemas (Troubleshooting)

### Erro: `ModuleNotFoundError`

- Confirme que a venv está ativada
- Reinstale dependências:

```bash
pip install -r requirements.txt
```

### Erro ao ler Excel

Garanta que `openpyxl` está instalado:

```bash
pip install openpyxl
```

### Treinamento muito lento

- Verifique GPU (seção 5)
- Reduza:
  - `--syn-public` e `--syn-nonpublic`
  - batch size no código (`train_batch_size`, `eval_batch_size`)
  - número de épocas (`num_train_epochs`)

### Colunas não detectadas

Use overrides:

```bash
python train.py --data "data/AMOSTRA_e-SIC.xlsx" --text-col "SUA_COLUNA_TEXTO" --label-col "SUA_COLUNA_LABEL"
```

---

## 10) Encerrando a venv

Quando terminar:

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
python train.py --data "data/AMOSTRA_e-SIC.xlsx" --output "runs/best_model"
```

### Linux/macOS

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train.py --data "data/AMOSTRA_e-SIC.xlsx" --output "runs/best_model"
```

## 👥 Equipe e Contexto do Projeto

Este projeto está sendo desenvolvido como parte de uma iniciativa de **inovação e experimentação tecnológica aplicada à gestão pública**, com foco no apoio à **triagem e análise de pedidos de acesso à informação (e-SIC)** por meio de técnicas de **Inteligência Artificial**.

O objetivo é explorar soluções práticas que auxiliem analistas e gestores públicos na tomada de decisão, promovendo maior eficiência, padronização e apoio técnico ao processo.

### Equipe de Desenvolvimento

- **Maikon Santos** — Desenvolvedor Fullstack  
  GitHub: [@Maikon-sant](https://github.com/Maikon-sant)

- **Maysa Santos** — Tech Lead & Desenvolvedora Fullstack  
  GitHub: [@Maysamkt](https://github.com/Maysamkt)
