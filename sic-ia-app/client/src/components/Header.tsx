/**
 * Header - SIC-IA
 * Design: Minimalismo Funcional com Tipografia Forte
 * Exibe título, subtítulo e aviso discreto sobre decisão final do gestor
 */

export default function Header() {
  return (
    <header className="border-b border-border bg-background py-8 px-4">
      <div className="container max-w-6xl mx-auto">
        {/* Título Principal */}
        <h1 className="text-4xl font-bold text-foreground mb-2">
          SIC-IA
        </h1>

        {/* Subtítulo */}
        <p className="text-lg text-muted-foreground mb-4">
          Classificação Inteligente de Pedidos e-SIC
        </p>

        {/* Aviso Discreto */}
        <div className="flex items-start gap-2 text-sm text-muted-foreground bg-secondary/30 border border-border rounded-md p-3 w-fit">
          <svg
            className="w-4 h-4 mt-0.5 flex-shrink-0"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
            />
          </svg>
          <span>
            Resultado sugerido pelo modelo. A decisão final é do gestor.
          </span>
        </div>
      </div>
    </header>
  );
}
