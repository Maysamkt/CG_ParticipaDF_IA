/**
 * Tipos TypeScript para SIC-IA
 * Aplicação de triagem de pedidos e-SIC com classificação por IA
 */

/**
 * Resposta da API para classificação de um texto individual
 */
export interface PredictResponse {
  label: "publico" | "nao_publico";
  score_nao_publico: number;
  threshold: number;
}

/**
 * Item individual de resultado de classificação
 */
export interface ClassificationResult {
  index: number;
  texto: string;
  label: "publico" | "nao_publico";
  score_nao_publico: number;
}

/**
 * Resposta da API para classificação em lote (Excel/CSV)
 */
export interface PredictExcelResponse {
  resultados: ClassificationResult[];
  qtd_publico: number;
  qtd_nao_publico: number;
  total: number;
  threshold: number;
  coluna_texto: string;
  filename: string;
}

/**
 * Estado de uma requisição assíncrona
 */
export type AsyncState<T> = 
  | { status: "idle" }
  | { status: "loading" }
  | { status: "success"; data: T }
  | { status: "error"; error: string };

/**
 * Configuração de filtros para tabela de resultados
 */
export interface TableFilters {
  searchText: string;
  classification: "todos" | "publico" | "nao_publico";
}

/**
 * Paginação para tabela de resultados
 */
export interface Pagination {
  page: number;
  pageSize: number;
}
