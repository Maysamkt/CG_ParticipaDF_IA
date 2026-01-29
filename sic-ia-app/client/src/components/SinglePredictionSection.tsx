/**
 * SinglePredictionSection - Seção de avaliação de pedido individual
 * Design: Minimalismo Funcional com Tipografia Forte
 * Textarea grande, campo de threshold, botão de avaliação, resultado em card
 */

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { AlertCircle, Copy, Trash2 } from "lucide-react";
import { toast } from "sonner";
import ClassificationBadge from "./ClassificationBadge";
import { predictSingleText } from "@/api/sicApi";
import type { PredictResponse, AsyncState } from "@/types";

export default function SinglePredictionSection() {
  const [texto, setTexto] = useState("");
  const [threshold, setThreshold] = useState<string>("");
  const [result, setResult] = useState<AsyncState<PredictResponse>>({
    status: "idle",
  });

  const isLoading = result.status === "loading";
  const isError = result.status === "error";
  const isSuccess = result.status === "success";

  const handleAvaliacao = async () => {
    // Validações
    if (!texto.trim()) {
      toast.error("Campo obrigatório", {
        description: "Digite o texto do pedido para avaliar.",
      });
      return;
    }

    if (texto.trim().length < 10) {
      toast.warning("Texto muito curto", {
        description: "Digite pelo menos 10 caracteres para uma avaliação confiável.",
      });
    }

    const thresholdValue = threshold ? parseFloat(threshold) : undefined;
    if (thresholdValue !== undefined && (thresholdValue < 0 || thresholdValue > 1)) {
      toast.error("Threshold inválido", {
        description: "Digite um valor entre 0 e 1.",
      });
      return;
    }

    setResult({ status: "loading" });

    try {
      const response = await predictSingleText(texto, thresholdValue);
      setResult({ status: "success", data: response });
      toast.success("Avaliação concluída", {
        description: `Classificação: ${response.label === "publico" ? "Público" : "Não Público"}`,
      });
    } catch (error) {
      const errorMessage =
        error instanceof Error ? error.message : "Erro desconhecido";
      setResult({ status: "error", error: errorMessage });
      toast.error("Erro na avaliação", {
        description: errorMessage,
      });
    }
  };

  const handleLimpar = () => {
    setTexto("");
    setThreshold("");
    setResult({ status: "idle" });
  };

  const handleCopiarResultado = () => {
    if (result.status === "success") {
      const jsonString = JSON.stringify(result.data, null, 2);
      navigator.clipboard.writeText(jsonString);
      toast.success("Resultado copiado", {
        description: "JSON do resultado foi copiado para a área de transferência.",
      });
    }
  };

  return (
    <div className="space-y-6">
      {/* Textarea */}
      <div className="space-y-2">
        <Label htmlFor="texto-pedido" className="text-base font-semibold">
          Texto do pedido
        </Label>
        <Textarea
          id="texto-pedido"
          placeholder="Cole aqui o texto completo do pedido de acesso à informação. Ex: Solicito acesso ao relatório de despesas do mês de janeiro..."
          value={texto}
          onChange={(e) => setTexto(e.target.value)}
          className="min-h-32 resize-none"
          disabled={isLoading}
          aria-label="Texto do pedido para avaliação"
        />
        <p className="text-xs text-muted-foreground">
          {texto.length} caracteres
        </p>
      </div>

      {/* Threshold */}
      <div className="space-y-2">
        <Label htmlFor="threshold" className="text-base font-semibold">
          Threshold (opcional)
        </Label>
        <Input
          id="threshold"
          type="number"
          min="0"
          max="1"
          step="0.01"
          placeholder="0.5"
          value={threshold}
          onChange={(e) => setThreshold(e.target.value)}
          disabled={isLoading}
          aria-label="Threshold de classificação"
        />
        <p className="text-xs text-muted-foreground">
          Quanto menor, mais rigoroso para marcar como Não Público. Padrão: 0.5
        </p>
      </div>

      {/* Botões de Ação */}
      <div className="flex gap-3">
        <Button
          onClick={handleAvaliacao}
          disabled={isLoading || !texto.trim()}
          size="lg"
          className="flex-1"
        >
          {isLoading ? (
            <>
              <svg
                className="w-4 h-4 mr-2 animate-spin"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M12 4v16m8-8H4"
                />
              </svg>
              Avaliando...
            </>
          ) : (
            "Avaliar"
          )}
        </Button>
        <Button
          onClick={handleLimpar}
          variant="outline"
          size="lg"
          disabled={isLoading}
          aria-label="Limpar formulário"
        >
          <Trash2 className="w-4 h-4" />
        </Button>
      </div>

      {/* Resultado */}
      {isSuccess && result.data && (
        <Card className="border-l-4 border-l-blue-500 bg-blue-50">
          <CardHeader>
            <CardTitle className="text-lg">Resultado da Avaliação</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-center gap-3">
              <span className="font-semibold">Classificação:</span>
              <ClassificationBadge label={result.data.label} size="lg" />
            </div>

            <div className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <p className="text-muted-foreground">Score de risco (Não Público)</p>
                <p className="text-lg font-bold text-foreground">
                  {(result.data.score_nao_publico * 100).toFixed(2)}%
                </p>
              </div>
              <div>
                <p className="text-muted-foreground">Threshold usado</p>
                <p className="text-lg font-bold text-foreground">
                  {(result.data.threshold * 100).toFixed(2)}%
                </p>
              </div>
            </div>

            <Button
              onClick={handleCopiarResultado}
              variant="outline"
              size="sm"
              className="w-full"
            >
              <Copy className="w-4 h-4 mr-2" />
              Copiar resultado (JSON)
            </Button>
          </CardContent>
        </Card>
      )}

      {/* Erro */}
      {isError && result.error && (
        <Card className="border-l-4 border-l-red-500 bg-red-50">
          <CardContent className="pt-6">
            <div className="flex gap-3">
              <AlertCircle className="w-5 h-5 text-red-600 flex-shrink-0 mt-0.5" />
              <div>
                <p className="font-semibold text-red-900">Erro na avaliação</p>
                <p className="text-sm text-red-800 mt-1">{result.error}</p>
                <p className="text-xs text-red-700 mt-2">
                  Verifique se a API está rodando em {import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000"}
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
