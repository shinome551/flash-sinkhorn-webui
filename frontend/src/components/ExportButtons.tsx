import { useState } from 'react'

interface Props {
  disabled: boolean
  onJson: () => void
  onPng: () => Promise<void>
}

const buttonClass =
  'rounded bg-slate-800 px-3 py-1 text-xs font-medium text-slate-200 ring-1 ring-slate-700 hover:bg-slate-700 disabled:cursor-not-allowed disabled:opacity-40'

/** 結果 JSON と可視化 PNG の保存。PNG は画像の読み込みを伴うので進行中・失敗を表示する。 */
export function ExportButtons({ disabled, onJson, onPng }: Props) {
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const savePng = async () => {
    setBusy(true)
    setError(null)
    try {
      await onPng()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="flex flex-wrap items-center gap-2">
      <span className="text-xs text-slate-400">保存:</span>
      <button type="button" className={buttonClass} disabled={disabled} onClick={onJson}>
        結果 JSON
      </button>
      <button type="button" className={buttonClass} disabled={disabled || busy} onClick={() => void savePng()}>
        {busy ? '生成中…' : '可視化 PNG'}
      </button>
      {error && (
        <span className="text-xs text-red-400" role="alert">
          {error}
        </span>
      )}
    </div>
  )
}
