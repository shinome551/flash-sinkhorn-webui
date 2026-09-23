// 結果 JSON のダウンロードと、可視化 PNG (画像 A・B + ヒートマップ / 矢印 / 接続線) の書き出し。

import { imageUrl } from '../api/client.ts'
import type { ImageUploadResponse, MatchRequest, MatchResponse } from '../api/types.ts'
import type { Assignment } from './assignment.ts'
import { viridisCss } from './colormap.ts'
import { drawFlow, drawGrid, drawHeatmap, drawOverlay } from './draw.ts'
import { buildFlow, FLOW_LABEL } from './flow.ts'
import { formatRange } from './format.ts'
import { buildHeatmap, heatmapLayer } from './heatmap.ts'
import { layoutLegend, type LegendLayout } from './legend.ts'
import { canPin, heatmapKind, showsTargets, type VizMode } from './modes.ts'
import { buildOverlay, type WeightNorm } from './overlay.ts'
import type { Pin } from './selection.ts'

/** 再現できるよう、実行に使ったリクエストと応答をまとめて 1 つの JSON にする。 */
export function resultJson(request: MatchRequest, response: MatchResponse): string {
  return JSON.stringify({ request, response }, null, 2)
}

export function downloadBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  a.remove()
  // ダウンロード開始前に解放しないよう、1 tick 遅らせる
  setTimeout(() => URL.revokeObjectURL(url), 0)
}

function loadImage(imageId: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image()
    img.onload = () => resolve(img)
    img.onerror = () => reject(new Error('画像を読み込めませんでした (保存期限切れの可能性があります)'))
    img.src = imageUrl(imageId)
  })
}

export interface PngInput {
  imageA: ImageUploadResponse
  imageB: ImageUploadResponse
  /** 表示用の結果 (lib/assignment.ts の viewResult 済み) */
  result: MatchResponse
  assignment: Assignment
  mode: VizMode
  pins: readonly Pin[]
  weightNorm: WeightNorm
  flowStep: number
  flowFade: boolean
  showGrid: boolean
}

/** 画像座標 1 ピクセルあたりの出力ピクセル数。画面 (最大 2 倍表示) と同程度の精細さにする。 */
const SCALE = 2
/** 画像 A と B の間隔 (画像ピクセル)。接続線の見える余白。 */
const GAP = 48
/** 凡例の帯 1 段の高さ (画像ピクセル)。幅が狭いと 2 段になる。 */
const CAPTION = 32
const BACKGROUND = '#0f172a'

/**
 * 画面と同じ図形で、A・B を左右に並べた PNG を作る。hover は一時的な状態なので含めない。
 * 凡例 (色バーと値の範囲) があるモードでは、下に帯を足す。
 */
export async function renderPng(input: PngInput): Promise<Blob> {
  const { imageA, imageB, result, assignment, mode, pins, weightNorm, flowStep, flowFade, showGrid } = input
  const [bitmapA, bitmapB] = await Promise.all([loadImage(imageA.image_id), loadImage(imageB.image_id)])

  const kind = heatmapKind(mode)
  const heatmap = kind ? buildHeatmap(result, kind, assignment) : null
  const geomA = { x: 0, y: 0, scale: SCALE }
  const geomB = { x: (imageA.width + GAP) * SCALE, y: 0, scale: SCALE }
  const flow = mode === 'flow' ? buildFlow(result, geomA, flowStep, flowFade) : null
  const legend = heatmap
    ? { label: heatmap.label, lo: heatmap.lo, hi: heatmap.hi }
    : flow
      ? { label: FLOW_LABEL, lo: flow.lo, hi: flow.hi }
      : null

  const canvas = document.createElement('canvas')
  const ctx = canvas.getContext('2d')
  if (!ctx) throw new Error('canvas を作成できませんでした')
  const imagesHeight = Math.max(imageA.height, imageB.height)
  const width = (imageA.width + GAP + imageB.width) * SCALE
  // 凡例の段数で高さが変わるので、文字幅を測ってから canvas の寸法を決める
  const legendLayout = legend
    ? layoutLegend(width, CAPTION * SCALE, legendTexts(legend), (text, size) => {
        ctx.font = legendFont(size)
        return ctx.measureText(text).width
      })
    : null
  canvas.width = width
  canvas.height = imagesHeight * SCALE + (legendLayout ? legendLayout.rows * CAPTION * SCALE : 0)
  ctx.fillStyle = BACKGROUND
  ctx.fillRect(0, 0, canvas.width, canvas.height)
  ctx.imageSmoothingQuality = 'high'

  // 画像・格子・ヒートマップは画像座標で描く。
  const inImageCoords = (geom: typeof geomA, image: HTMLImageElement, draw: () => void) => {
    ctx.save()
    ctx.translate(geom.x, geom.y)
    ctx.drawImage(image, 0, 0, image.naturalWidth * SCALE, image.naturalHeight * SCALE)
    ctx.scale(SCALE, SCALE)
    draw()
    ctx.restore()
  }
  inImageCoords(geomA, bitmapA, () => {
    if (showGrid) drawGrid(ctx, result.grid_a, 1 / SCALE)
    const layerA = heatmapLayer(heatmap, 'a')
    if (heatmap && layerA) drawHeatmap(ctx, heatmap, layerA)
  })
  inImageCoords(geomB, bitmapB, () => {
    if (showGrid) drawGrid(ctx, result.grid_b, 1 / SCALE)
    const layerB = heatmapLayer(heatmap, 'b')
    if (heatmap && layerB) drawHeatmap(ctx, heatmap, layerB)
  })

  // 図形はステージ座標 (= 出力ピクセル) で描く。
  drawOverlay(
    ctx,
    buildOverlay({
      result,
      a: geomA,
      b: geomB,
      pins: canPin(mode) ? pins : [],
      active: null,
      weightNorm,
      showTargets: showsTargets(mode),
    }),
  )
  if (flow) drawFlow(ctx, flow)

  if (legend && legendLayout) drawLegend(ctx, legendTexts(legend), legendLayout, imagesHeight * SCALE)

  return new Promise((resolve, reject) => {
    canvas.toBlob((blob) => (blob ? resolve(blob) : reject(new Error('PNG を生成できませんでした'))), 'image/png')
  })
}

const legendFont = (size: number) => `${size}px sans-serif`

function legendTexts(legend: { label: string; lo: number; hi: number }) {
  const [lo, hi] = formatRange(legend.lo, legend.hi)
  return { label: legend.label, lo, hi }
}

function drawLegend(
  ctx: CanvasRenderingContext2D,
  texts: { label: string; lo: string; hi: string },
  layout: LegendLayout,
  top: number,
): void {
  const { bar } = layout
  ctx.save()
  ctx.translate(0, top)
  ctx.font = legendFont(layout.fontSize)
  ctx.textBaseline = 'middle'
  ctx.fillStyle = '#cbd5e1'
  ctx.fillText(texts.label, layout.label.x, layout.label.y)

  const gradient = ctx.createLinearGradient(bar.x, 0, bar.x + bar.width, 0)
  for (let k = 0; k <= 8; k++) gradient.addColorStop(k / 8, viridisCss(k / 8))
  ctx.fillStyle = gradient
  ctx.fillRect(bar.x, bar.y, bar.width, bar.height)
  ctx.fillStyle = '#cbd5e1'
  ctx.textAlign = 'right'
  ctx.fillText(texts.lo, layout.lo.x, layout.lo.y)
  ctx.textAlign = 'left'
  ctx.fillText(texts.hi, layout.hi.x, layout.hi.y)
  ctx.restore()
}
