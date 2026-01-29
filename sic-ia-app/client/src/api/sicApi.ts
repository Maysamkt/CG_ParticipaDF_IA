/**
 * Funções de API para SIC-IA
 * Comunicação com backend de classificação de pedidos e-SIC
 */

import type { PredictResponse, PredictExcelResponse } from "@/types";

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

/**
 * Classificar um texto individual
 * @param texto - Texto do pedido a ser classificado
 * @param threshold - Limiar de classificação (opcional)
 * @returns Resultado da classificação
 */
export async function predictSingleText(
  texto: string,
  threshold?: number
): Promise<PredictResponse> {
  const payload: Record<string, unknown> = { texto };
  if (threshold !== undefined) {
    payload.threshold = threshold;
  }

  const response = await fetch(`${API_BASE_URL}/predict`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(
      errorData.detail ||
        `Erro na API: ${response.status} ${response.statusText}`
    );
  }

  return response.json();
}

/**
 * Classificar arquivo Excel/CSV em lote
 * @param file - Arquivo a ser processado
 * @param colunaTexto - Nome da coluna contendo os textos
 * @param threshold - Limiar de classificação (opcional)
 * @param batchSize - Tamanho do lote (opcional)
 * @returns Resultados da classificação em lote
 */
export async function predictExcelFile(
  file: File,
  colunaTexto: string,
  threshold?: number,
  batchSize?: number
): Promise<PredictExcelResponse> {
  const coluna = colunaTexto.trim();

  if (!coluna) {
    throw new Error("Digite o nome da coluna de texto.");
  }

  const formData = new FormData();
  formData.append("file", file);

  const params = new URLSearchParams();
  params.set("coluna_texto", coluna);

  if (threshold !== undefined) params.set("threshold", String(threshold));
  if (batchSize !== undefined) params.set("batch_size", String(batchSize));

  const response = await fetch(
    `${API_BASE_URL}/predict-excel?${params.toString()}`,
    {
      method: "POST",
      body: formData,
    }
  );

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(
      errorData.detail ||
        `Erro na API: ${response.status} ${response.statusText}`
    );
  }

  return response.json();
}

/**
 * Validar se a API está acessível
 * @returns true se API está disponível, false caso contrário
 */
export async function checkApiHealth(): Promise<boolean> {
  try {
    const response = await fetch(`${API_BASE_URL}/`, {
      method: "GET",
    });
    return response.ok;
  } catch {
    return false;
  }
}
