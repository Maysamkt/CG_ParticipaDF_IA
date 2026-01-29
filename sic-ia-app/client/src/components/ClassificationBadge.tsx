/**
 * ClassificationBadge - Badge de classificação
 * Exibe "Público" (verde) ou "Não Público" (vermelho)
 */

interface ClassificationBadgeProps {
  label: "publico" | "nao_publico";
  size?: "sm" | "md" | "lg";
}

export default function ClassificationBadge({
  label,
  size = "md",
}: ClassificationBadgeProps) {
  const isPublico = label === "publico";

  const sizeClasses = {
    sm: "px-2 py-1 text-xs font-semibold",
    md: "px-3 py-1.5 text-sm font-semibold",
    lg: "px-4 py-2 text-base font-bold",
  };

  const colorClasses = isPublico
    ? "bg-green-100 text-green-800 border border-green-300"
    : "bg-red-100 text-red-800 border border-red-300";

  const displayText = isPublico ? "Público" : "Não Público";

  return (
    <span
      className={`inline-flex items-center rounded-md ${sizeClasses[size]} ${colorClasses}`}
    >
      {isPublico ? (
        <svg className="w-4 h-4 mr-1" fill="currentColor" viewBox="0 0 20 20">
          <path
            fillRule="evenodd"
            d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z"
            clipRule="evenodd"
          />
        </svg>
      ) : (
        <svg className="w-4 h-4 mr-1" fill="currentColor" viewBox="0 0 20 20">
          <path
            fillRule="evenodd"
            d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z"
            clipRule="evenodd"
          />
        </svg>
      )}
      {displayText}
    </span>
  );
}
