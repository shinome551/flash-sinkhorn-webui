import { memo, useEffect, useRef } from 'react'
import { drawHeatmap } from '../lib/draw.ts'
import { heatmapLayer, type Heatmap } from '../lib/heatmap.ts'

/**
 * 画像の上に重ねるヒートマップ (canvas) のうち `side` の画像の分。そのレイヤが無ければ何も出さない。入力は通さない。
 * 値と表示倍率が変わったときだけ描き直す (hover では再描画しない)。
 */
export const HeatmapLayer = memo(function HeatmapLayer({
  heatmap,
  side,
  displayScale,
}: {
  heatmap: Heatmap | null
  side: 'a' | 'b'
  displayScale: number
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const layer = heatmapLayer(heatmap, side)
  const cssWidth = (layer?.grid.image_width ?? 0) * displayScale
  const cssHeight = (layer?.grid.image_height ?? 0) * displayScale

  useEffect(() => {
    const canvas = canvasRef.current
    const ctx = canvas?.getContext('2d')
    if (!canvas || !ctx || !heatmap || !layer) return
    const dpr = window.devicePixelRatio || 1
    canvas.width = Math.round(cssWidth * dpr)
    canvas.height = Math.round(cssHeight * dpr)
    ctx.scale(dpr * displayScale, dpr * displayScale)
    drawHeatmap(ctx, heatmap, layer)
  }, [heatmap, layer, displayScale, cssWidth, cssHeight])

  if (!layer) return null

  return (
    <canvas
      ref={canvasRef}
      className="pointer-events-none absolute left-0 top-0 rounded"
      style={{ width: cssWidth, height: cssHeight }}
      aria-hidden
    />
  )
})
