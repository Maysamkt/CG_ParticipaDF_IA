/**
 * Home - Página principal SIC-IA
 * Design: Minimalismo Funcional com Tipografia Forte
 * Abas para pedido individual e avaliação em lote
 */

import { useState } from "react";
import Header from "@/components/Header";
import SinglePredictionSection from "@/components/SinglePredictionSection";
import BatchPredictionSection from "@/components/BatchPredictionSection";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

export default function Home() {
  const [activeTab, setActiveTab] = useState("individual");

  return (
    <div className="min-h-screen flex flex-col bg-background">
      <Header />

      <main className="flex-1 py-8 px-4">
        <div className="container max-w-6xl mx-auto">
          <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
            <TabsList className="grid w-full max-w-md grid-cols-2 mb-8">
              <TabsTrigger value="individual">Pedido Individual</TabsTrigger>
              <TabsTrigger value="lote">Avaliação em Lote</TabsTrigger>
            </TabsList>

            <TabsContent value="individual" className="space-y-6">
              <div>
                <h2 className="text-2xl font-bold text-foreground mb-2">
                  Avaliação de Pedido Individual
                </h2>
                <p className="text-muted-foreground">
                  Classifique um pedido de acesso à informação como Público ou Não Público
                </p>
              </div>
              <SinglePredictionSection />
            </TabsContent>

            <TabsContent value="lote" className="space-y-6">
              <div>
                <h2 className="text-2xl font-bold text-foreground mb-2">
                  Avaliação em Lote
                </h2>
                <p className="text-muted-foreground">
                  Carregue uma planilha (Excel ou CSV) para classificar múltiplos pedidos de uma vez
                </p>
              </div>
              <BatchPredictionSection />
            </TabsContent>
          </Tabs>
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-border bg-secondary/30 py-6 px-4 text-center text-sm text-muted-foreground">
        <p>
          SIC-IA © 2026 | Ferramenta de apoio à triagem de pedidos e-SIC
        </p>
      </footer>
    </div>
  );
}
