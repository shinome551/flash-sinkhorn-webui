import { useMutation } from '@tanstack/react-query'
import { useCallback, useEffect, useRef, useState } from 'react'
import { postMatch } from '../api/client.ts'
import type { MatchRequest, MatchResponse, MatchStats } from '../api/types.ts'
import { isAbortError } from '../lib/errors.ts'

interface Done {
  request: MatchRequest
  response: MatchResponse
}

/**
 * マッチングの実行。新しく実行すると進行中のリクエストは中断され、結果は常に最新の 1 件だけを持つ。
 * `cancel` は中断して結果とエラーを消す (中断ボタン)。`reset` はさらに実行履歴も捨てる (入力画像を差し替えたとき)。
 * `request` は表示中の結果を得たときのリクエスト、`previousStats` はその 1 つ前の成功した実行の統計
 * (`reset` までの間。再実行中に結果が消えても保持する。パラメータ変更の前後比較用)。
 * 注意: 中断は fetch を abort するだけで、サーバ側の計算は完走する (GPU セマフォ内のスレッドは止められない)。
 */
export function useMatch(options: { onError?: (error: unknown, req: MatchRequest) => void } = {}) {
  const abortRef = useRef<AbortController | null>(null)
  const onErrorRef = useRef(options.onError)
  useEffect(() => {
    onErrorRef.current = options.onError
  })

  const [history, setHistory] = useState<{ latest: Done | null; previous: Done | null }>({
    latest: null,
    previous: null,
  })

  const mutation = useMutation({
    mutationFn: async (request: MatchRequest): Promise<Done> => {
      abortRef.current?.abort()
      const controller = new AbortController()
      abortRef.current = controller
      return { request, response: await postMatch(request, controller.signal) }
    },
    onSuccess: (done) => setHistory((h) => ({ latest: done, previous: h.latest })),
    // 中断 (AbortError) は失敗として扱わない
    onError: (error, req) => {
      if (!isAbortError(error)) onErrorRef.current?.(error, req)
    },
  })
  const { mutate, reset: resetMutation } = mutation

  useEffect(() => () => abortRef.current?.abort(), [])

  const run = useCallback((req: MatchRequest) => mutate(req), [mutate])

  const cancel = useCallback(() => {
    abortRef.current?.abort()
    resetMutation()
  }, [resetMutation])

  const reset = useCallback(() => {
    cancel()
    setHistory({ latest: null, previous: null })
  }, [cancel])

  const previousStats: MatchStats | null = history.previous?.response.stats ?? null

  return {
    run,
    cancel,
    reset,
    isPending: mutation.isPending,
    result: mutation.data?.response ?? null,
    request: mutation.data?.request ?? null,
    previousStats,
    error: mutation.error && !isAbortError(mutation.error) ? mutation.error : null,
  }
}
