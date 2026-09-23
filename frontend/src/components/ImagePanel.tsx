import { useEffect, useRef, useState, type ReactNode, type Ref } from 'react'
import type { GridInfo, ImageUploadResponse } from '../api/types.ts'
import { imageUrl } from '../api/client.ts'
import { drawGrid } from '../lib/draw.ts'
import { patchCount } from '../lib/grid.ts'

/** これ以上には拡大しない (前処理後の画像は最大 512px 程度なので、広い画面でぼけすぎないように)。 */
const MAX_DISPLAY_SCALE = 2

interface Props {
  image: ImageUploadResponse
  /** 指定すると格子を重ねて描画する。座標系は image の寸法 (前処理後) */
  grid?: GridInfo | null
  /** 画像座標→表示座標の倍率 (displayScale) を受け取って、canvas の上に重ねる要素を返す */
  children?: (displayScale: number) => ReactNode
  /** canvas と重ね要素を包む box (画像と同寸法) の ref。接続線のために、親が位置を測るのに使う */
  boxRef?: Ref<HTMLDivElement>
  /** 画像を取得できなかった (TTL 切れなど) */
  onLoadError?: () => void
}

/**
 * 前処理後の画像を canvas に描画する。canvas の論理座標は画像ピクセルと一致し、
 * 表示は幅に合わせた `displayScale` 倍 (CSS ピクセル) になる。
 */
export function ImagePanel({ image, grid, children, boxRef, onLoadError }: Props) {
  const containerRef = useRef<HTMLDivElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [bitmap, setBitmap] = useState<HTMLImageElement | null>(null)
  const [containerWidth, setContainerWidth] = useState(0)
  const onLoadErrorRef = useRef(onLoadError)
  useEffect(() => {
    onLoadErrorRef.current = onLoadError
  })

  useEffect(() => {
    const img = new Image()
    let cancelled = false
    img.onload = () => {
      if (!cancelled) setBitmap(img)
    }
    img.onerror = () => {
      if (!cancelled) onLoadErrorRef.current?.()
    }
    img.src = imageUrl(image.image_id)
    return () => {
      cancelled = true
      setBitmap(null)
    }
  }, [image.image_id])

  useEffect(() => {
    const el = containerRef.current
    if (!el) return
    const observer = new ResizeObserver(([entry]) => setContainerWidth(entry.contentRect.width))
    observer.observe(el)
    return () => observer.disconnect()
  }, [])

  const displayScale = containerWidth > 0 ? Math.min(containerWidth / image.width, MAX_DISPLAY_SCALE) : 0
  const cssWidth = image.width * displayScale
  const cssHeight = image.height * displayScale

  useEffect(() => {
    const canvas = canvasRef.current
    const ctx = canvas?.getContext('2d')
    if (!canvas || !ctx || !bitmap || displayScale === 0) return

    const dpr = window.devicePixelRatio || 1
    canvas.width = Math.round(cssWidth * dpr)
    canvas.height = Math.round(cssHeight * dpr)
    ctx.imageSmoothingQuality = 'high'
    ctx.drawImage(bitmap, 0, 0, canvas.width, canvas.height)

    if (grid && patchCount(grid) > 0) {
      // 画像座標で描く。1 単位 = displayScale CSS px なので、線幅 1/displayScale で 1 CSS px になる。
      ctx.scale(dpr * displayScale, dpr * displayScale)
      drawGrid(ctx, grid, 1 / displayScale)
    }
  }, [bitmap, grid, displayScale, cssWidth, cssHeight])

  return (
    <div ref={containerRef} className="w-full">
      <div ref={boxRef} className="relative mx-auto" style={{ width: cssWidth, height: cssHeight }}>
        <canvas
          ref={canvasRef}
          className="block rounded"
          style={{ width: cssWidth, height: cssHeight }}
          aria-label={`画像 (${image.width}×${image.height})`}
        />
        {displayScale > 0 && children?.(displayScale)}
      </div>
    </div>
  )
}
