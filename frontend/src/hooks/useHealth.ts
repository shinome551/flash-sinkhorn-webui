import { useQuery } from '@tanstack/react-query'
import { getHealth } from '../api/client.ts'

/** /api/health。サーバ停止からの復帰を拾うため定期的に再取得する。 */
export function useHealth() {
  return useQuery({
    queryKey: ['health'],
    queryFn: ({ signal }) => getHealth(signal),
    refetchInterval: 15_000,
    retry: false,
  })
}
