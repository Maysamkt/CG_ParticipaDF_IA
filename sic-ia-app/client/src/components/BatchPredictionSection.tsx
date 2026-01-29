/**
 * BatchPredictionSection - Seção de avaliação em lote (Excel/CSV)
 * Design: Minimalismo Funcional com Tipografia Forte
 * Upload com drag-and-drop, campos de configuração, tabela de resultados
 */

import { useState, useRef } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { AlertCircle, Upload, Download, Search } from "lucide-react";
import { toast } from "sonner";
import ClassificationBadge from "./ClassificationBadge";
import { predictExcelFile } from "@/api/sicApi";
import type {
  PredictExcelResponse,
  ClassificationResult,
  AsyncState,
  TableFilters,
  Pagination,
} from "@/types";

export default function BatchPredictionSection() {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [colunaTexto, setColunaTexto] = useState("");
  const [threshold, setThreshold] = useState<string>("");
  const [batchSize, setBatchSize] = useState("32");
  const [result, setResult] = useState<AsyncState<PredictExcelResponse>>({
    status: "idle",
  });
  const [filters, setFilters] = useState<TableFilters>({
    searchText: "",
    classification: "todos",
  });
  const [pagination, setPagination] = useState<Pagination>({
    page: 0,
    pageSize: 10,
  });

  const isLoading = result.status === "loading";
  const isError = result.status === "error";
  const isSuccess = result.status === "success";

  // Filtrar resultados
  const getFilteredResults = (): ClassificationResult[] => {
    if (result.status !== "success") return [];

    return result.data.resultados.filter(item => {
      const matchesSearch = item.texto
        .toLowerCase()
        .includes(filters.searchText.toLowerCase());
      const matchesClassification =
        filters.classification === "todos" ||
        item.label === filters.classification;
      return matchesSearch && matchesClassification;
    });
  };

  const filteredResults = getFilteredResults();
  const totalPages = Math.ceil(filteredResults.length / pagination.pageSize);
  const paginatedResults = filteredResults.slice(
    pagination.page * pagination.pageSize,
    (pagination.page + 1) * pagination.pageSize
  );

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      handleFileSelect(files[0]);
    }
  };

  const handleFileSelect = (selectedFile: File) => {
    const validTypes = [
      "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
      "application/vnd.ms-excel",
      "text/csv",
    ];

    if (!validTypes.includes(selectedFile.type)) {
      toast.error("Arquivo inválido", {
        description: "Envie apenas arquivos .xlsx ou .csv",
      });
      return;
    }

    setFile(selectedFile);
    toast.success("Arquivo selecionado", {
      description: `${selectedFile.name} (${(selectedFile.size / 1024).toFixed(2)} KB)`,
    });
  };

  const handleAvaliacao = async () => {
    if (!file) {
      toast.error("Campo obrigatório", {
        description: "Selecione um arquivo para avaliar.",
      });
      return;
    }

    if (!colunaTexto.trim()) {
      toast.error("Campo obrigatório", {
        description: "Digite o nome da coluna de texto.",
      });
      return;
    }

    const thresholdValue = threshold ? parseFloat(threshold) : undefined;
    if (
      thresholdValue !== undefined &&
      (thresholdValue < 0 || thresholdValue > 1)
    ) {
      toast.error("Threshold inválido", {
        description: "Digite um valor entre 0 e 1.",
      });
      return;
    }

    const batchSizeValue = parseInt(batchSize) || 32;

    setResult({ status: "loading" });
    setPagination({ page: 0, pageSize: 10 });

    try {
      const response = await predictExcelFile(
        file,
        colunaTexto,
        thresholdValue,
        batchSizeValue
      );
      setResult({ status: "success", data: response });
      toast.success("Avaliação concluída", {
        description: `${response.total} pedidos processados`,
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

  const handleExportarCSV = () => {
    if (result.status !== "success") return;

    const headers = ["Index", "Texto", "Classificação", "Score"];
    const rows = result.data.resultados.map(item => [
      item.index,
      `"${item.texto.replace(/"/g, '""')}"`,
      item.label === "publico" ? "Público" : "Não Público",
      (item.score_nao_publico * 100).toFixed(2),
    ]);

    const csv = [headers, ...rows].map(row => row.join(",")).join("\n");
    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `sic-ia-resultados-${new Date().toISOString().split("T")[0]}.csv`;
    a.click();
    URL.revokeObjectURL(url);

    toast.success("Arquivo exportado", {
      description: "Resultados foram salvos em CSV",
    });
  };

  return (
    <div className="space-y-6">
      {/* Upload Area */}
      <div
        className="border-2 border-dashed border-border rounded-lg p-8 text-center cursor-pointer hover:bg-secondary/30 transition-colors"
        onDragOver={handleDragOver}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
        role="button"
        tabIndex={0}
        onKeyDown={e => {
          if (e.key === "Enter" || e.key === " ") {
            fileInputRef.current?.click();
          }
        }}
        aria-label="Área de upload de arquivo"
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".xlsx,.csv"
          onChange={e => {
            if (e.target.files?.[0]) {
              handleFileSelect(e.target.files[0]);
            }
          }}
          className="hidden"
          aria-label="Selecionar arquivo"
        />
        <Upload className="w-8 h-8 mx-auto mb-2 text-muted-foreground" />
        <p className="font-semibold text-foreground">
          Arraste um arquivo aqui ou clique para selecionar
        </p>
        <p className="text-sm text-muted-foreground mt-1">
          Aceita .xlsx e .csv
        </p>
        {file && (
          <p className="text-sm text-green-600 mt-2 font-medium">
            ✓ {file.name} selecionado
          </p>
        )}
      </div>

      {/* Configurações */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="space-y-2">
          <Label htmlFor="coluna-texto" className="text-base font-semibold">
            Nome da coluna de texto
          </Label>
          <Input
            id="coluna-texto"
            placeholder="Ex.: Texto Mascarado"
            value={colunaTexto}
            onChange={e => setColunaTexto(e.target.value)}
            disabled={isLoading}
            aria-label="Nome da coluna de texto"
          />
        </div>

        <div className="space-y-2">
          <Label htmlFor="threshold-batch" className="text-base font-semibold">
            Threshold (opcional)
          </Label>
          <Input
            id="threshold-batch"
            type="number"
            min="0"
            max="1"
            step="0.01"
            placeholder="0.5"
            value={threshold}
            onChange={e => setThreshold(e.target.value)}
            disabled={isLoading}
            aria-label="Threshold de classificação"
          />
        </div>

        <div className="space-y-2">
          <Label htmlFor="batch-size" className="text-base font-semibold">
            Batch size
          </Label>
          <Input
            id="batch-size"
            type="number"
            min="1"
            max="1000"
            placeholder="32"
            value={batchSize}
            onChange={e => setBatchSize(e.target.value)}
            disabled={isLoading}
            aria-label="Tamanho do lote"
          />
        </div>
      </div>

      {/* Botão de Avaliação */}
      <Button
        onClick={handleAvaliacao}
        disabled={isLoading || !file}
        size="lg"
        className="w-full"
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
            Avaliando planilha...
          </>
        ) : (
          "Avaliar planilha"
        )}
      </Button>

      {/* Resumo */}
      {isSuccess && result.data && (
        <>
          <div className="grid grid-cols-3 gap-4">
            <Card>
              <CardContent className="pt-6 text-center">
                <p className="text-sm text-muted-foreground mb-1">Total</p>
                <p className="text-3xl font-bold text-foreground">
                  {result.data.total}
                </p>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="pt-6 text-center">
                <p className="text-sm text-muted-foreground mb-1">Públicos</p>
                <p className="text-3xl font-bold text-green-600">
                  {result.data.qtd_publico}
                </p>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="pt-6 text-center">
                <p className="text-sm text-muted-foreground mb-1">
                  Não Públicos
                </p>
                <p className="text-3xl font-bold text-red-600">
                  {result.data.qtd_nao_publico}
                </p>
              </CardContent>
            </Card>
          </div>

          {/* Filtros e Busca */}
          <div className="space-y-4">
            <div className="flex gap-3">
              <div className="flex-1 relative">
                <Search className="absolute left-3 top-3 w-4 h-4 text-muted-foreground" />
                <Input
                  placeholder="Buscar por texto..."
                  value={filters.searchText}
                  onChange={e => {
                    setFilters({ ...filters, searchText: e.target.value });
                    setPagination({ page: 0, pageSize: 10 });
                  }}
                  className="pl-10"
                  aria-label="Buscar resultados"
                />
              </div>
              <select
                value={filters.classification}
                onChange={e => {
                  setFilters({
                    ...filters,
                    classification: e.target.value as any,
                  });
                  setPagination({ page: 0, pageSize: 10 });
                }}
                className="px-3 py-2 border border-border rounded-md text-sm bg-background"
                aria-label="Filtrar por classificação"
              >
                <option value="todos">Todos</option>
                <option value="publico">Público</option>
                <option value="nao_publico">Não Público</option>
              </select>
            </div>

            {/* Tabela */}
            <div className="border border-border rounded-lg overflow-hidden">
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead className="bg-secondary border-b border-border">
                    <tr>
                      <th className="px-4 py-3 text-left font-semibold">
                        Index
                      </th>
                      <th className="px-4 py-3 text-left font-semibold">
                        Texto
                      </th>
                      <th className="px-4 py-3 text-left font-semibold">
                        Classificação
                      </th>
                      <th className="px-4 py-3 text-right font-semibold">
                        Score
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {paginatedResults.length > 0 ? (
                      paginatedResults.map((item, idx) => (
                        <tr
                          key={`${item.index}-${idx}`}
                          className="border-b border-border hover:bg-secondary/50 transition-colors"
                        >
                          <td className="px-4 py-3 text-foreground font-medium">
                            {item.index}
                          </td>
                          <td className="px-4 py-3 text-foreground max-w-xs truncate">
                            {item.texto}
                          </td>
                          <td className="px-4 py-3">
                            <ClassificationBadge label={item.label} size="sm" />
                          </td>
                          <td className="px-4 py-3 text-right font-medium">
                            {(item.score_nao_publico * 100).toFixed(2)}%
                          </td>
                        </tr>
                      ))
                    ) : (
                      <tr>
                        <td
                          colSpan={4}
                          className="px-4 py-8 text-center text-muted-foreground"
                        >
                          Nenhum resultado encontrado
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Paginação */}
            {totalPages > 1 && (
              <div className="flex items-center justify-between">
                <p className="text-sm text-muted-foreground">
                  Mostrando {pagination.page * pagination.pageSize + 1} a{" "}
                  {Math.min(
                    (pagination.page + 1) * pagination.pageSize,
                    filteredResults.length
                  )}{" "}
                  de {filteredResults.length} resultados
                </p>
                <div className="flex gap-2">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() =>
                      setPagination({
                        ...pagination,
                        page: Math.max(0, pagination.page - 1),
                      })
                    }
                    disabled={pagination.page === 0}
                  >
                    Anterior
                  </Button>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() =>
                      setPagination({
                        ...pagination,
                        page: Math.min(totalPages - 1, pagination.page + 1),
                      })
                    }
                    disabled={pagination.page >= totalPages - 1}
                  >
                    Próximo
                  </Button>
                </div>
              </div>
            )}

            {/* Exportar */}
            <Button
              onClick={handleExportarCSV}
              variant="outline"
              className="w-full"
            >
              <Download className="w-4 h-4 mr-2" />
              Exportar como CSV
            </Button>
          </div>
        </>
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
                  Verifique se a API está rodando em{" "}
                  {import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000"}
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
