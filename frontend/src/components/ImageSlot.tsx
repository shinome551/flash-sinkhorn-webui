import { useState, type ReactNode, type Ref } from 'react'
import type { GridInfo, ImageUploadResponse } from '../api/types.ts'
import { useImageUpload } from '../hooks/useImageUpload.ts'
import { describeError } from '../lib/errors.ts'
import { ImageDropzone } from './ImageDropzone.tsx'
import { ImagePanel } from './ImagePanel.tsx'

interface Props {
  label: string
  image: ImageUploadResponse | null
  onChange: (image: ImageUploadResponse | null) => void
  grid?: GridInfo | null
  maxBytes?: number
  /** ImagePanel の box の ref と、その上に重ねる要素 (displayScale を受け取る) */
  boxRef?: Ref<HTMLDivElement>
  overlay?: (displayScale: number) => ReactNode
  /** 親側で期限切れを検知した (マッチ実行時の IMAGE_NOT_FOUND など)。画像は外した上で通知だけ出す */
  expired?: boolean
}

const EXPIRED_NOTICE = {
  title: '画像が見つかりません (保存期限切れの可能性があります)',
  detail: '再度アップロードしてください。',
  hint: null,
}

/** 1 枚分の入力欄。未選択ならドロップゾーン、選択済みなら canvas 表示。 */
export function ImageSlot({
  label,
  image,
  onChange,
  grid,
  maxBytes,
  boxRef,
  overlay,
  expired: expiredByParent,
}: Props) {
  const [expiredHere, setExpired] = useState(false)
  // 親側で画像が入った (サンプル読み込みなど) ら、以前の期限切れの案内は捨てる (render 中の state 調整)
  if (image && expiredHere) setExpired(false)
  const expired = expiredHere || expiredByParent
  const uploader = useImageUpload((uploaded) => {
    setExpired(false)
    onChange(uploaded)
  })

  const error = uploader.error ? describeError(uploader.error) : expired ? EXPIRED_NOTICE : null

  return (
    <section className="min-w-0">
      <div className="mb-2 flex items-baseline justify-between gap-2">
        <h2 className="text-sm font-semibold text-slate-200">{label}</h2>
        {image && (
          <div className="flex items-baseline gap-3 text-xs text-slate-400">
            <span>
              {image.width}×{image.height}
              {(image.original_width !== image.width || image.original_height !== image.height) &&
                ` (元 ${image.original_width}×${image.original_height})`}
            </span>
            <button type="button" onClick={() => onChange(null)} className="underline hover:text-slate-200">
              クリア
            </button>
          </div>
        )}
      </div>
      {image ? (
        <ImagePanel
          image={image}
          grid={grid}
          boxRef={boxRef}
          onLoadError={() => {
            setExpired(true)
            onChange(null)
          }}
        >
          {overlay}
        </ImagePanel>
      ) : (
        <ImageDropzone
          label={label}
          onFile={uploader.upload}
          previewUrl={uploader.previewUrl}
          progress={uploader.isUploading ? uploader.progress : null}
          onCancel={uploader.cancel}
          error={error}
          maxBytes={maxBytes}
        />
      )}
    </section>
  )
}
