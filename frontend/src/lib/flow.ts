// flow モード: A 側パッチ中心から、対応先の重心への変位 (`displacement`, ピクセル) を矢印にする。
// 座標は overlay.ts と同じくステージ内の CSS ピクセル。描画は MatchOverlay / draw.ts。

import type { MatchResponse, PatchMatch } from '../api/types.ts'
import { viridisScaled } from './colormap.ts'
import { colorRange } from './heatmap.ts'
import { isUnbalanced, massOpacity } from './mass.ts'
import { patchDisplayCenter, toIndex, type Point, type Rect } from './grid.ts'
import type { BoxGeometry } from './overlay.ts'

export interface FlowArrow {
  key: string
  x1: number
  y1: number
  x2: number
  y2: number
  color: string
  /** 確信度・送出量で薄くするときの不透明度 (MIN_FLOW_OPACITY..1)。薄くしないときは 1 */
  opacity: number
  /** 矢じりの 3 頂点。短すぎる矢印 (長さ < MIN_ARROW_LENGTH) は null で、点として描く */
  head: [Point, Point, Point] | null
}

export interface Flow {
  arrows: FlowArrow[]
  /** 色の下端・上端に対応する変位の大きさ [px] (画像座標) */
  lo: number
  hi: number
  /** 矢印を切り取る範囲 (A の画像領域)。変位は A の外まで伸びうる */
  clip: Rect
}

/** 矢印の色の凡例 (画面・PNG 共通) */
export const FLOW_LABEL = 'A 側パッチの変位の大きさ [px]'

/** 確信度が最も低い矢印の不透明度。0 にすると対応の無いパッチが消えて、格子の抜けに見える。 */
export const MIN_FLOW_OPACITY = 0.12

/**
 * 確信度 → 不透明度。色範囲と同じ分位点 (両端 2% を飽和) で [0, 1] に写してから MIN_FLOW_OPACITY..1 へ。
 * 対応先の無い端のパッチは拡散した分布の重心を指して長い矢印になるが、確信度が低いので目立たなくなる。
 */
export function confidenceOpacity(confidences: readonly number[]): (confidence: number) => number {
  const [lo, hi] = colorRange(confidences)
  if (!(hi > lo)) return () => 1
  return (c) => MIN_FLOW_OPACITY + (1 - MIN_FLOW_OPACITY) * Math.min(1, Math.max(0, (c - lo) / (hi - lo)))
}

/** これより短い (表示 px) 矢印は矢じりを描かず点にする。 */
export const MIN_ARROW_LENGTH = 4

/** 矢じり。先端 (x2, y2) から始点側へ `size` 戻った点を軸に、±`size/2` 広げる。 */
export function arrowHead(x1: number, y1: number, x2: number, y2: number, size: number): [Point, Point, Point] {
  const length = Math.hypot(x2 - x1, y2 - y1)
  const ux = (x2 - x1) / length
  const uy = (y2 - y1) / length
  const bx = x2 - ux * size
  const by = y2 - uy * size
  return [
    { x: x2, y: y2 },
    { x: bx - uy * (size / 2), y: by + ux * (size / 2) },
    { x: bx + uy * (size / 2), y: by - ux * (size / 2) },
  ]
}

/**
 * `step` パッチおきに (行・列とも) 間引いた矢印を作る。色は変位の大きさを viridis に写す。
 * 色範囲は間引き前の全パッチから決めるので、密度を変えても色は変わらない。
 * `fadeByConfidence` なら確信度の低い矢印ほど薄くする (confidenceOpacity)。unbalanced の結果では、これとは別に
 * 送出量 (row_mass) の小さい矢印も薄くする。質量をほとんど送らないパッチの変位は、対応として意味を持たないため。
 */
export function buildFlow(result: MatchResponse, a: BoxGeometry, step: number, fadeByConfidence = false): Flow {
  const { grid_a: grid, matches } = result
  const magnitudes = matches.map((m) => Math.hypot(m.displacement[0], m.displacement[1]))
  const [lo, hi] = colorRange(magnitudes)
  const stride = Math.max(1, Math.round(step))
  const byConfidence = fadeByConfidence ? confidenceOpacity(matches.map((m) => m.confidence)) : () => 1
  const byMass = isUnbalanced(result) ? massOpacity : () => 1
  const opacity = (m: PatchMatch) =>
    Math.max(MIN_FLOW_OPACITY, Math.min(byConfidence(m.confidence), byMass(m.row_mass)))

  const arrows: FlowArrow[] = []
  for (let row = 0; row < grid.rows; row += stride) {
    for (let col = 0; col < grid.cols; col += stride) {
      const i = toIndex(grid, row, col)
      const match = matches[i]
      if (!match) continue
      const from = patchDisplayCenter(grid, i, a.scale)
      const x1 = a.x + from.x
      const y1 = a.y + from.y
      const x2 = x1 + match.displacement[0] * a.scale
      const y2 = y1 + match.displacement[1] * a.scale
      const length = Math.hypot(x2 - x1, y2 - y1)
      arrows.push({
        key: `flow${i}`,
        x1,
        y1,
        x2,
        y2,
        color: viridisScaled(magnitudes[i], lo, hi),
        opacity: opacity(match),
        head: length >= MIN_ARROW_LENGTH ? arrowHead(x1, y1, x2, y2, Math.min(7, length * 0.5)) : null,
      })
    }
  }

  return {
    arrows,
    lo,
    hi,
    clip: { x: a.x, y: a.y, width: grid.image_width * a.scale, height: grid.image_height * a.scale },
  }
}
