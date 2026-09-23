// Canvas 2D への描画ヘルパ。画面上のヒートマップ / 格子と、PNG 書き出し (export.ts) で同じ図形を共用する。
// 画面上の接続線・矢印は SVG (MatchOverlay) で描くが、図形の生成は overlay.ts / flow.ts で共通。

import type { GridInfo } from '../api/types.ts'
import { viridisScaled } from './colormap.ts'
import type { Flow } from './flow.ts'
import { patchCount, patchRect } from './grid.ts'
import type { Heatmap, HeatmapLayer } from './heatmap.ts'
import { HALO_COLOR, type Overlay } from './overlay.ts'

/** ヒートマップの不透明度。画像の模様が透けて見える程度。 */
export const HEATMAP_ALPHA = 0.6

/** パッチ格子の線を描く。ctx は画像座標 (1 単位 = 1 画像ピクセル) に scale 済みであること。 */
export function drawGrid(ctx: CanvasRenderingContext2D, grid: GridInfo, lineWidth: number): void {
  const n = patchCount(grid)
  if (n <= 0) return
  ctx.save()
  ctx.lineWidth = lineWidth
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.35)'
  ctx.beginPath()
  for (let i = 0; i < n; i++) {
    const r = patchRect(grid, i)
    ctx.rect(r.x, r.y, r.width, r.height)
  }
  ctx.stroke()
  ctx.restore()
}

/** ヒートマップの 1 レイヤを塗る。色範囲は `heatmap` のもの。ctx は画像座標に scale 済みであること。 */
export function drawHeatmap(ctx: CanvasRenderingContext2D, heatmap: Heatmap, layer: HeatmapLayer): void {
  ctx.save()
  ctx.globalAlpha = HEATMAP_ALPHA
  for (let i = 0; i < layer.values.length; i++) {
    const r = patchRect(layer.grid, i)
    ctx.fillStyle = viridisScaled(layer.values[i], heatmap.lo, heatmap.hi)
    ctx.fillRect(r.x, r.y, r.width, r.height)
  }
  ctx.restore()
}

/** hover / pin の枠と接続線を描く。ctx はステージ座標 (CSS ピクセル相当) であること。 */
export function drawOverlay(ctx: CanvasRenderingContext2D, overlay: Overlay): void {
  ctx.save()
  for (const r of [...overlay.rectsA, ...overlay.rectsB]) {
    ctx.fillStyle = r.fill
    ctx.globalAlpha = r.fillOpacity
    ctx.fillRect(r.x, r.y, r.width, r.height)
    ctx.strokeStyle = r.stroke
    ctx.globalAlpha = r.strokeOpacity
    ctx.lineWidth = r.strokeWidth
    ctx.strokeRect(r.x, r.y, r.width, r.height)
  }
  ctx.lineCap = 'round'
  for (const l of overlay.lines) {
    ctx.globalAlpha = l.opacity
    for (const [color, width] of [
      [HALO_COLOR, l.width + 2],
      [l.color, l.width],
    ] as const) {
      ctx.strokeStyle = color
      ctx.lineWidth = width
      ctx.beginPath()
      ctx.moveTo(l.x1, l.y1)
      ctx.lineTo(l.x2, l.y2)
      ctx.stroke()
    }
    ctx.beginPath()
    ctx.arc(l.x2, l.y2, l.width / 2 + 1.5, 0, Math.PI * 2)
    ctx.fillStyle = l.color
    ctx.fill()
    ctx.strokeStyle = HALO_COLOR
    ctx.lineWidth = 1
    ctx.stroke()
  }
  ctx.restore()
}

/** flow の矢印を描く。A の画像領域 (`flow.clip`) の外は切り取る。ctx はステージ座標であること。 */
export function drawFlow(ctx: CanvasRenderingContext2D, flow: Flow): void {
  ctx.save()
  ctx.beginPath()
  ctx.rect(flow.clip.x, flow.clip.y, flow.clip.width, flow.clip.height)
  ctx.clip()
  ctx.lineCap = 'round'
  ctx.lineJoin = 'round'
  for (const f of flow.arrows) {
    ctx.globalAlpha = f.opacity
    if (f.head === null) {
      ctx.beginPath()
      ctx.arc(f.x1, f.y1, 1.5, 0, Math.PI * 2)
      ctx.fillStyle = f.color
      ctx.fill()
      continue
    }
    for (const [color, width] of [
      [HALO_COLOR, 3.5],
      [f.color, 1.5],
    ] as const) {
      ctx.strokeStyle = color
      ctx.lineWidth = width
      ctx.beginPath()
      ctx.moveTo(f.x1, f.y1)
      ctx.lineTo(f.x2, f.y2)
      ctx.stroke()
    }
    ctx.beginPath()
    ctx.moveTo(f.head[0].x, f.head[0].y)
    ctx.lineTo(f.head[1].x, f.head[1].y)
    ctx.lineTo(f.head[2].x, f.head[2].y)
    ctx.closePath()
    ctx.fillStyle = f.color
    ctx.fill()
    ctx.strokeStyle = HALO_COLOR
    ctx.lineWidth = 1
    ctx.stroke()
  }
  ctx.restore()
}
