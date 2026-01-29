import { useRef } from "react";

type noop = (...args: any[]) => any;

/**
 * usePersistFn é uma alternativa ao useCallback
 * para reduzir carga cognitiva e evitar problemas com closures.
 *
 * Ele retorna uma função com referência estável,
 * mas que sempre executa a implementação mais recente.
 */
export function usePersistFn<T extends noop>(fn: T) {
  // Guarda sempre a versão mais recente da função
  const fnRef = useRef<T>(fn);
  fnRef.current = fn;

  // Função persistente (referência nunca muda)
  const persistFn = useRef<T>(null);

  if (!persistFn.current) {
    persistFn.current = function (this: unknown, ...args) {
      // Executa a função mais recente armazenada no ref
      return fnRef.current!.apply(this, args);
    } as T;
  }

  return persistFn.current!;
}
