import { useHealth } from '../hooks/useHealth.ts'
import { describeError } from '../lib/errors.ts'

const BASE = 'inline-flex items-center gap-2 rounded-full px-3 py-1 text-xs font-medium ring-1'

export function HealthBadge() {
  const { data, error, isPending } = useHealth()

  if (isPending) {
    return (
      <span className={`${BASE} bg-slate-800 text-slate-400 ring-slate-700`} role="status">
        <Dot className="bg-slate-500" />
        接続中…
      </span>
    )
  }

  // 再取得に失敗したときは、直前の値ではなく「停止中」を出す
  if (error || !data) {
    const { title, hint } = describeError(error)
    return (
      <span
        className={`${BASE} bg-red-950 text-red-300 ring-red-800`}
        role="status"
        title={[title, hint].filter(Boolean).join('\n')}
      >
        <Dot className="bg-red-500" />
        サーバ停止中
      </span>
    )
  }

  const tooltip = [
    `device: ${data.device}`,
    `torch ${data.torch_version}`,
    `flash-sinkhorn ${data.flash_sinkhorn_version ?? '未導入'}`,
    `上限: ${data.limits.max_patches} パッチ / ${(data.limits.max_upload_bytes / 1024 / 1024).toFixed(0)} MB`,
  ].join('\n')

  return data.cuda_available ? (
    <span className={`${BASE} bg-emerald-950 text-emerald-300 ring-emerald-800`} role="status" title={tooltip}>
      <Dot className="bg-emerald-500" />
      {data.gpu_name ?? 'GPU'} · {data.default_backend}
    </span>
  ) : (
    <span className={`${BASE} bg-amber-950 text-amber-300 ring-amber-800`} role="status" title={tooltip}>
      <Dot className="bg-amber-500" />
      GPU なし (CPU) · {data.default_backend}
    </span>
  )
}

function Dot({ className }: { className: string }) {
  return <span className={`h-2 w-2 rounded-full ${className}`} aria-hidden />
}
