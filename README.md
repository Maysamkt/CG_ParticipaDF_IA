# CG_ParticipaDF_IA

Classificação Inteligente de Pedidos e-SIC com Inteligência Artificial

Este repositório reúne todo o ecossistema do projeto SIC-IA, desenvolvido como uma iniciativa de inovação tecnológica aplicada à gestão pública, com foco no apoio à triagem e análise de pedidos de acesso à informação (e-SIC) por meio de Inteligência Artificial.

O projeto integra:

- Treinamento e experimentação de modelos de IA
- Uma API de inferência pronta para uso institucional
- Uma aplicação web para uso por analistas e gestores públicos

## 🎯 Objetivo do Projeto

Desenvolver e demonstrar uma solução completa que auxilie o poder público a:

- Classificar automaticamente pedidos e-SIC como Público ou Não Público
- Apoiar analistas na triagem inicial de pedidos
- Reduzir esforço manual e aumentar padronização
- Explorar o uso prático de IA em fluxos reais de gestão pública

## 🧱 Estrutura Geral do Repositório

```bash
CG_ParticipaDF_IA/
├── modelo/          # Treinamento, experimentos e artefatos do modelo de IA
├── api/             # API FastAPI para inferência do modelo
├── sic-ia-app/      # Aplicação web (frontend)
└── README.md        # Este arquivo
```

## 📦 Descrição das Pastas

### 📁 modelo/

Contém scripts, notebooks e artefatos relacionados ao treinamento e experimentação do modelo de IA.  
Esta pasta documenta o processo técnico e não é necessária para executar a aplicação final.

### 📁 api/

API desenvolvida em FastAPI responsável por carregar o modelo treinado e disponibilizar endpoints de inferência.

Consulte o README interno para instruções completas de uso.

### 📁 sic-ia-app/

Aplicação web desenvolvida em React para interação com a API, avaliação individual e em lote, visualização e exportação de resultados.

Consulte o README interno para detalhes de execução.

## 👥 Equipe de Desenvolvimento

- **Maikon Santos** — Desenvolvedor Fullstack  
  GitHub: https://github.com/Maikon-sant

- **Maysa Santos** — Tech Lead & Desenvolvedora Fullstack  
  GitHub: https://github.com/Maysamkt

## 📄 Licença

MIT
