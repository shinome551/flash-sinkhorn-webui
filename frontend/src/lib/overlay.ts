// マッチ結果と選択状態から、2 画像をまたぐ SVG オーバーレイの図形を作る (描画そのものは MatchOverlay)。
// 座標はすべて「ステージ」(2 枚の画像を含む共通の親要素) 内の CSS ピクセル。

import type { MatchResponse } from '../api/types.ts'
import { viridisCss } from './colormap.ts'
import { patchDisplayCenter, patchDisplayRect } from './grid.ts'
import { isUnbalanced, massOpacity } from './mass.ts'
import type { Pin } from './selection.ts'

/** ステージ座標での画像の左上と、画像座標→表示座標の倍率 (displayScale)。 */
export interface BoxGeometry {
  x: number
  y: number
  scale: number
}

/** pin の識別色 (Okabe-Ito から、写真の上でも見分けやすい 7 色)。slot が余ったら循環する。 */
export const PIN_COLORS = ['#e69f00', '#56b4e9', '#009e73', '#f0e442', '#0072b2', '#d55e00', '#cc79a7']

/** hover 中のパッチ (A 側) の枠色。 */
export const HOVER_COLOR = '#ffffff'

/** 接続線・矢印の下に敷く暗い縁取り。明るい画像の上でも読めるようにする。 */
export const HALO_COLOR = 'rgba(0, 0, 0, 0.55)'

/**
 * 重みの見た目上の基準。absolute = 条件付き確率そのまま、relative = 各パッチの最大の重み (top-1) を 1 とする。
 * 既定パラメータでは分布が拡散して絶対値が小さい (0.03〜0.08 程度) ので、relative なら線が薄くなりすぎない。
 */
export type WeightNorm = 'absolute' | 'relative'

export function pinColor(slot: number): string {
  return PIN_COLORS[slot % PIN_COLORS.length]
}

const clamp01 = (v: number) => Math.min(Math.max(Number.isNaN(v) ? 0 : v, 0), 1)

/** 重み (条件付き確率) の見た目上の強さ 0..1。小さい重みも見えるよう平方根を取る。 */
function weightLevel(weight: number): number {
  return Math.sqrt(clamp01(weight))
}

/** 重み → 不透明度。0 に近い重みでも 0.15 は残す (top-k に入っているものが消えないように)。 */
export function weightOpacity(weight: number): number {
  return 0.15 + 0.85 * weightLevel(weight)
}

/** 重み → 接続線の太さ (CSS px)。 */
export function weightLineWidth(weight: number): number {
  return 0.75 + 3.25 * weightLevel(weight)
}

/** hover 時の重み → 色 (viridis)。暗い紫の端は画像に埋もれるので 0.3..1 の範囲だけ使う。 */
export function weightColor(weight: number): string {
  return viridisCss(0.3 + 0.7 * weightLevel(weight))
}

export interface OverlayRect {
  key: string
  x: number
  y: number
  width: number
  height: number
  fill: string
  fillOpacity: number
  stroke: string
  strokeOpacity: number
  strokeWidth: number
}

export interface OverlayLine {
  key: string
  x1: number
  y1: number
  x2: number
  y2: number
  color: string
  width: number
  opacity: number
}

export interface Overlay {
  rectsA: OverlayRect[]
  rectsB: OverlayRect[]
  lines: OverlayLine[]
}

export interface OverlayInput {
  result: MatchResponse
  a: BoxGeometry
  b: BoxGeometry
  pins: readonly Pin[]
  /** hover / 矢印キーで指している A 側パッチ */
  active: number | null
  /** 既定 absolute */
  weightNorm?: WeightNorm
  /** false なら hover の対応先 (B 側の枠と接続線) を描かず、A 側の枠だけにする。既定 true */
  showTargets?: boolean
}

/**
 * pin は識別色 (重みは不透明度と太さで表す)、hover は重みを viridis に写した色で描く。
 * unbalanced の結果では、pin の対応先を送出量 (row_mass) で薄くする (重みは条件付き確率なので、送出量が 0 に
 * 近いパッチでも濃く見えてしまう)。hover は HoverInfo に送出量の数値を出すので薄くしない。
 * 描画順は pin → hover で、hover が最前面になる。pin 済みのパッチを hover しても、
 * 対応先は pin 側で描いているので A 側の枠だけを重ねる。
 */
export function buildOverlay({
  result,
  a,
  b,
  pins,
  active,
  weightNorm = 'absolute',
  showTargets = true,
}: OverlayInput): Overlay {
  const { grid_a: gridA, grid_b: gridB, matches } = result
  const rectsA: OverlayRect[] = []
  const rectsB: OverlayRect[] = []
  const lines: OverlayLine[] = []

  const rectA = (key: string, i: number, color: string, fillOpacity: number): OverlayRect => {
    const r = patchDisplayRect(gridA, i, a.scale)
    return {
      key,
      x: a.x + r.x,
      y: a.y + r.y,
      width: r.width,
      height: r.height,
      fill: color,
      fillOpacity,
      stroke: color,
      strokeOpacity: 1,
      strokeWidth: 2,
    }
  }

  const unbalanced = isUnbalanced(result)

  /** i の対応先を B 側の矩形と接続線として追加する。色は `colorOf(weight)`、不透明度は `fade` 倍。 */
  const addTargets = (prefix: string, i: number, colorOf: (weight: number) => string, fade = 1) => {
    const match = matches[i]
    if (!match) return
    const from = patchDisplayCenter(gridA, i, a.scale)
    // relative は top-1 が 1 になるよう割る。targets は weight 降順だが、順序に依存せず最大を取る。
    const top = weightNorm === 'relative' ? Math.max(...match.targets.map((t) => t.weight)) : 1
    for (const t of match.targets) {
      const weight = top > 0 ? t.weight / top : 0
      const color = colorOf(weight)
      const opacity = weightOpacity(weight) * fade
      const r = patchDisplayRect(gridB, t.j, b.scale)
      rectsB.push({
        key: `${prefix}-B${t.j}`,
        x: b.x + r.x,
        y: b.y + r.y,
        width: r.width,
        height: r.height,
        fill: color,
        fillOpacity: opacity * 0.6,
        stroke: color,
        strokeOpacity: Math.min(1, opacity + 0.2),
        strokeWidth: 1.5,
      })
      const to = patchDisplayCenter(gridB, t.j, b.scale)
      lines.push({
        key: `${prefix}-L${t.j}`,
        x1: a.x + from.x,
        y1: a.y + from.y,
        x2: b.x + to.x,
        y2: b.y + to.y,
        color,
        width: weightLineWidth(weight),
        opacity,
      })
    }
  }

  for (const pin of pins) {
    if (!matches[pin.i]) continue
    const color = pinColor(pin.slot)
    rectsA.push(rectA(`pin${pin.i}-A`, pin.i, color, 0.25))
    addTargets(`pin${pin.i}`, pin.i, () => color, unbalanced ? massOpacity(matches[pin.i].row_mass) : 1)
  }

  if (active !== null && matches[active]) {
    if (pins.some((p) => p.i === active)) {
      rectsA.push({ ...rectA('hover-A', active, HOVER_COLOR, 0), strokeWidth: 3 })
    } else {
      rectsA.push(rectA('hover-A', active, HOVER_COLOR, 0.2))
      if (showTargets) addTargets('hover', active, weightColor)
    }
  }

  return { rectsA, rectsB, lines }
}
