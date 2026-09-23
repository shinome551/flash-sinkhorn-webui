import { useHealth } from '../hooks/useHealth.ts'

/**
 * サーバ停止中 / GPU なしの案内。ヘッダーのバッジだけでは気づきにくいので、本文の先頭に出す。
 * 停止中は起動コマンドと再接続ボタン、GPU なしは CPU で動くときの制約を示す。
 */
export function ServerNotice() {
  const { data, error, refetch, isFetching } = useHealth()

  if (error) {
    return (
      <div className="rounded border border-red-800 bg-red-950/60 px-3 py-2 text-sm text-red-300" role="alert">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <p className="font-medium">バックエンドに接続できません</p>
          <button
            type="button"
            onClick={() => void refetch()}
            disabled={isFetching}
            className="rounded bg-red-900/60 px-3 py-1 text-xs text-red-100 hover:bg-red-900 disabled:opacity-60"
          >
            {isFetching ? '確認中…' : '再接続'}
          </button>
        </div>
        <p className="mt-1 text-xs text-red-300/80">
          <code className="rounded bg-slate-900 px-1">cd backend && uv run uvicorn app.main:app --port 8000</code>{' '}
          で起動してください。起動を確認すると自動で復帰します (15 秒ごとに確認)。
        </p>
      </div>
    )
  }

  if (data && !data.cuda_available) {
    return (
      <p className="rounded border border-amber-900 bg-amber-950/40 px-3 py-2 text-xs text-amber-300" role="status">
        GPU (CUDA) が見つからないため、CPU の dense バックエンドで計算します。パッチ数の上限は {data.limits.max_patches}{' '}
        で、計算に数秒以上かかることがあります。
      </p>
    )
  }
  return null
}
