# SIC-IA - Classificação Inteligente de Pedidos e-SIC

Aplicação web para apoiar analistas e gestores na triagem de pedidos de acesso à informação (e-SIC) usando classificação automática por IA.

## 🎯 Objetivo

Ferramenta interna simples e eficiente para classificar pedidos como **Público** ou **Não Público**, com suporte a:

- Avaliação individual de pedidos
- Processamento em lote de planilhas (Excel/CSV)
- Filtros, busca e exportação de resultados

## 🛠 Stack Técnico

- **Frontend:** React 19 + Vite + TypeScript
- **Styling:** Tailwind CSS 4 + shadcn/ui
- **HTTP Client:** Fetch API
- **Componentes:** shadcn/ui (Button, Card, Input, Textarea, Tabs, etc.)
- **Notificações:** Sonner (toast notifications)
- **Ícones:** Lucide React

## 📋 Requisitos

- Node.js 18+ (recomendado 22+)
- npm ou pnpm
- API backend rodando em `VITE_API_BASE_URL` (padrão: `http://127.0.0.1:8000`)

## 🚀 Como Rodar

### 1. Instalar dependências

```bash
npm install
# ou
pnpm install
```

### 2. Configurar variável de ambiente

Crie um arquivo `.env.local` na raiz do projeto:

```env
VITE_API_BASE_URL=http://127.0.0.1:8000
```

Se a API estiver em outro endereço, atualize a URL conforme necessário.

### 3. Iniciar servidor de desenvolvimento

```bash
npm run dev
# ou
pnpm dev
```

A aplicação estará disponível em `http://localhost:5173`

### 4. Build para produção

```bash
npm run build
# ou
pnpm build
```

## 📡 Endpoints da API

### POST /predict

Classificar um texto individual.

**Request:**

```json
{
  "texto": "Solicito acesso ao relatório de despesas do mês de janeiro",
  "threshold": 0.5
}
```

**Response:**

```json
{
  "label": "publico",
  "score_nao_publico": 0.23,
  "threshold": 0.5
}
```

### POST /predict-excel

Classificar arquivo em lote (multipart/form-data).

**Campos do formulário:**

- `file` (obrigatório): Arquivo .xlsx ou .csv
- `coluna_texto` (obrigatório no frontend): Nome da coluna que contém os textos a serem classificados.
  - O valor informado é enviado diretamente para a API.
  - Caso o nome esteja incorreto, a API retorna um erro informando as colunas disponíveis no arquivo.

- `threshold` (opcional): Limiar de classificação (0-1)
- `batch_size` (padrão: 32): Tamanho do lote de processamento

**Response:**

```json
{
  "resultados": [
    {
      "index": 0,
      "texto": "Solicito acesso...",
      "label": "publico",
      "score_nao_publico": 0.23
    }
  ],
  "qtd_publico": 45,
  "qtd_nao_publico": 55,
  "total": 100,
  "threshold": 0.5,
  "coluna_texto": "texto",
  "filename": "pedidos.xlsx"
}
```

## 🎨 Design

**Filosofia:** Minimalismo Funcional com Tipografia Forte

- **Cores:** Verde (#10b981) para Público, Vermelho (#ef4444) para Não Público, Azul (#3b82f6) para ações
- **Tipografia:** Poppins Bold para títulos, Inter Regular para corpo
- **Espaçamento:** Generoso, com respiração visual
- **Acessibilidade:** Labels, ARIA, foco visível, contraste adequado, navegação por teclado

## 🔧 Estrutura do Projeto

```bash
client/
├── public/              # Arquivos estáticos
├── src/
│   ├── api/            # Funções de chamada à API
│   │   └── sicApi.ts
│   ├── components/     # Componentes React reutilizáveis
│   │   ├── Header.tsx
│   │   ├── ClassificationBadge.tsx
│   │   ├── SinglePredictionSection.tsx
│   │   ├── BatchPredictionSection.tsx
│   │   └── ui/         # shadcn/ui components
│   ├── contexts/       # React contexts
│   ├── hooks/          # Custom hooks
│   ├── pages/          # Páginas (rotas)
│   │   └── Home.tsx
│   ├── types/          # TypeScript interfaces
│   │   └── index.ts
│   ├── App.tsx         # Componente raiz
│   ├── main.tsx        # Entry point
│   └── index.css       # Estilos globais
├── index.html          # Template HTML
└── package.json
```

## 📝 Exemplos de Uso

### Avaliação Individual

1. Acesse a aba "Pedido Individual"
2. Cole o texto do pedido no textarea
3. (Opcional) Defina um threshold customizado
4. Clique em "Avaliar"
5. Veja o resultado com badge de classificação e score
6. Use "Copiar resultado" para copiar o JSON

### Avaliação em Lote

1. Acesse a aba "Avaliação em Lote"
2. Arraste um arquivo .xlsx ou .csv, ou clique para selecionar
3. Configure:
   - **Nome da coluna de texto:** Nome exato da coluna da planilha que contém os textos dos pedidos.

- Este campo é obrigatório.
- Caso o nome não corresponda a nenhuma coluna, a API retorna uma mensagem informando as colunas existentes.

- **Threshold (opcional):** Limiar de classificação
- **Batch size:** Tamanho do lote (padrão: 32)

4. Clique em "Avaliar planilha"
5. Veja resumo com totais e tabela paginada de resultados
6. Use filtros e busca para navegar resultados
7. Clique em "Exportar como CSV" para baixar resultados

> 💡 **Observação importante**
>
> O frontend não tenta inferir automaticamente a coluna de texto.
> O nome informado pelo usuário é enviado diretamente para a API, que realiza a validação.
> Caso o valor esteja incorreto, a resposta da API informa quais colunas existem no arquivo,
> auxiliando o usuário a corrigir o preenchimento.

## ⚠️ Validações e Tratamento de Erros

- **Texto vazio:** Mensagem de erro abaixo do campo
- **Arquivo inválido:** Erro claro "Envie .xlsx ou .csv"
- **Coluna de texto vazia:** Bloqueio de envio no frontend
- **Coluna inexistente:** Erro retornado pela API com lista de colunas disponíveis

- **API offline:** Alerta com sugestão de verificação da URL
- **Threshold inválido:** Erro com intervalo válido (0-1)

## 🔐 Acessibilidade

- Labels associados a todos os campos
- ARIA labels em elementos interativos
- Foco visível em navegação por teclado
- Contraste adequado entre texto e fundo
- Navegação completa por teclado
- Estados de carregamento e erro comunicados

## 📊 Performance

- Tabela paginada (10 itens por página) para lotes grandes
- Filtros e busca executados localmente
- Re-renders otimizados com React hooks
- Lazy loading de componentes quando necessário

## 🐛 Troubleshooting

### "Erro na API: 404"

Verifique se a API está rodando em `VITE_API_BASE_URL`. Padrão: `http://127.0.0.1:8000`

### "Arquivo inválido"

Certifique-se de enviar apenas arquivos .xlsx ou .csv

### "Campo obrigatório"

Verifique se todos os campos obrigatórios foram preenchidos

## 📄 Licença

MIT

## 👥 Suporte

Para dúvidas ou problemas, entre em contato com a equipe de desenvolvimento.

## 👥 Equipe e Contexto do Projeto

Este projeto está sendo desenvolvido como parte de uma iniciativa de **inovação e experimentação tecnológica aplicada à gestão pública**, com foco no apoio à **triagem e análise de pedidos de acesso à informação (e-SIC)** por meio de técnicas de **Inteligência Artificial**.

O objetivo é explorar soluções práticas que auxiliem analistas e gestores públicos na tomada de decisão, promovendo maior eficiência, padronização e apoio técnico ao processo.

### Equipe de Desenvolvimento

- **Maikon Santos** — Desenvolvedor Fullstack  
  GitHub: [@Maikon-sant](https://github.com/Maikon-sant)

- **Maysa Santos** — Tech Lead & Desenvolvedora Fullstack  
  GitHub: [@Maysamkt](https://github.com/Maysamkt)
