import { useId, useState } from 'react'
import type { ErrorText } from '../lib/errors.ts'

const ACCEPTED_TYPES = ['image/png', 'image/jpeg', 'image/webp']

interface Props {
  label: string
  onFile: (file: File) => void
  /** アップロード中に表示するローカルプレビュー */
  previewUrl?: string | null
  /** 0..1。null / undefined ならアップロード中ではない */
  progress?: number | null
  onCancel?: () => void
  error?: ErrorText | null
  /** これを超えるファイルは送信せずに拒否する (health の limits.max_upload_bytes) */
  maxBytes?: number
}

export function ImageDropzone({ label, onFile, previewUrl, progress, onCancel, error, maxBytes }: Props) {
  const inputId = useId()
  const [dragging, setDragging] = useState(false)
  const [localError, setLocalError] = useState<string | null>(null)
  const uploading = progress != null

  const accept = (file: File | undefined) => {
    if (!file) return
    if (!ACCEPTED_TYPES.includes(file.type)) {
      setLocalError('PNG / JPEG / WebP の画像を選んでください。')
      return
    }
    if (maxBytes != null && file.size > maxBytes) {
      setLocalError(
        `ファイルが大きすぎます (${(file.size / 1024 / 1024).toFixed(1)} MB、上限 ${(maxBytes / 1024 / 1024).toFixed(0)} MB)。`,
      )
      return
    }
    setLocalError(null)
    onFile(file)
  }

  const shownError = localError ?? error?.title
  const detail = localError ? null : [error?.detail, error?.hint].filter(Boolean).join(' ')

  return (
    <div>
      <label
        htmlFor={inputId}
        onDragOver={(e) => {
          e.preventDefault()
          setDragging(true)
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault()
          setDragging(false)
          accept(e.dataTransfer.files[0])
        }}
        className={`relative flex min-h-64 cursor-pointer flex-col items-center justify-center gap-2 overflow-hidden rounded-lg border-2 border-dashed p-6 text-center transition-colors focus-within:ring-2 focus-within:ring-sky-500 ${
          dragging ? 'border-sky-400 bg-sky-950/40' : 'border-slate-600 bg-slate-800/40 hover:border-slate-500'
        }`}
      >
        {uploading && previewUrl && (
          <img src={previewUrl} alt="" className="absolute inset-0 h-full w-full object-contain opacity-30" />
        )}
        <input
          id={inputId}
          type="file"
          accept={ACCEPTED_TYPES.join(',')}
          className="sr-only"
          disabled={uploading}
          onChange={(e) => {
            accept(e.target.files?.[0])
            e.target.value = '' // 同じファイルをもう一度選べるようにする
          }}
        />
        {uploading ? (
          <div className="relative w-full max-w-xs" role="status">
            <p className="mb-2 text-sm text-slate-200">
              {progress < 1 ? `アップロード中… ${Math.round(progress * 100)}%` : '画像を処理中…'}
            </p>
            <div className="h-1.5 overflow-hidden rounded bg-slate-700">
              <div
                className={`h-full bg-sky-500 transition-[width] ${progress >= 1 ? 'animate-pulse' : ''}`}
                style={{ width: `${Math.round(progress * 100)}%` }}
              />
            </div>
          </div>
        ) : (
          <>
            <p className="text-sm font-medium text-slate-200">{label} をドロップ、またはクリックして選択</p>
            <p className="text-xs text-slate-500">PNG / JPEG / WebP</p>
          </>
        )}
      </label>
      {uploading && onCancel && (
        <button type="button" onClick={onCancel} className="mt-2 text-xs text-slate-400 underline hover:text-slate-200">
          キャンセル
        </button>
      )}
      {shownError && (
        <p className="mt-2 text-sm text-red-400" role="alert">
          {shownError}
          {detail && <span className="mt-0.5 block text-xs text-red-400/70">{detail}</span>}
        </p>
      )}
    </div>
  )
}
