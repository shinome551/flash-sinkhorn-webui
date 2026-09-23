// ヒートマップ (confidence / entropy は A 側、coverage は A・B 両側) の値と表示範囲。描画は draw.ts / HeatmapLayer。

import type { GridInfo, MatchResponse } from '../api/types.ts'
import type { Assignment } from './assignment.ts'
import type { HeatmapKind } from './modes.ts'

/** 1 枚の画像に塗る値 */
export interface HeatmapLayer {
  side: 'a' | 'b'
  grid: GridInfo
  /** パッチ i の値 (i = row * cols + col) */
  values: number[]
}

export interface Heatmap {
  kind: HeatmapKind
  /** 画像ごとの値。色範囲と凡例は全レイヤで共通 */
  layers: HeatmapLayer[]
  /** 色の下端・上端に対応する値。外れ値に引きずられないよう分位点で決める */
  lo: number
  hi: number
  label: string
}

/** `side` の画像に塗るレイヤ。無ければ undefined。 */
export function heatmapLayer(heatmap: Heatmap | null, side: 'a' | 'b'): HeatmapLayer | undefined {
  return heatmap?.layers.find((l) => l.side === side)
}

/** 昇順にした値の分位点 (最近傍)。`q` は 0..1。空なら 0。 */
export function quantile(sorted: readonly number[], q: number): number {
  if (sorted.length === 0) return 0
  const k = Math.round(Math.min(Math.max(q, 0), 1) * (sorted.length - 1))
  return sorted[k]
}

/** 色範囲 [lo, hi]。両端の 2% ずつを飽和させる (値が 1 点だけ突出しても全体が同じ色にならない)。 */
export function colorRange(values: readonly number[]): [number, number] {
  const sorted = values.filter(Number.isFinite).sort((p, q) => p - q)
  return [quantile(sorted, 0.02), quantile(sorted, 0.98)]
}

/**
 * coverage の色範囲が最低でも含む幅 (1.0 の上下)。balanced では col_mass が 1.0 の近傍 (1e-4 程度の
 * 数値誤差) に収まるので、データの範囲だけで色を決めると、意味のない誤差が強い色の模様になる。
 */
export const COVERAGE_MIN_HALF_SPAN = 0.1

/**
 * `result` は表示用 (lib/assignment.ts の viewResult 済み)。`assignment` は凡例の文言と coverage の A 側の有無に使う。
 * coverage は A = 送出量 (row_mass)、B = 受け取り量 (col_mass) を共通の色範囲で塗る。ハード割当では A の各パッチが
 * ちょうど 1 回ずつ送るので、B 側だけ。
 */
export function buildHeatmap(result: MatchResponse, kind: HeatmapKind, assignment: Assignment = 'soft'): Heatmap {
  const layer = (side: 'a' | 'b', values: number[]): HeatmapLayer => ({
    side,
    grid: side === 'a' ? result.grid_a : result.grid_b,
    values,
  })
  const make = (layers: HeatmapLayer[], label: string): Heatmap => {
    let [lo, hi] = colorRange(layers.flatMap((l) => l.values))
    if (kind === 'coverage') {
      lo = Math.min(lo, 1 - COVERAGE_MIN_HALF_SPAN)
      hi = Math.max(hi, 1 + COVERAGE_MIN_HALF_SPAN)
    }
    return { kind, layers, lo, hi, label }
  }
  switch (kind) {
    case 'confidence':
      return make(
        [
          layer(
            'a',
            result.matches.map((m) => m.confidence),
          ),
        ],
        '確信度 (最大の条件付き確率)',
      )
    case 'entropy':
      return make(
        [
          layer(
            'a',
            result.matches.map((m) => m.entropy),
          ),
        ],
        '行エントロピー [nat]',
      )
    case 'coverage':
      return assignment === 'hard'
        ? make([layer('b', result.col_mass)], 'ハード割当の受け取り量 (選ばれた回数 × M/N、1.0 = 均等)')
        : make(
            [
              layer(
                'a',
                result.matches.map((m) => m.row_mass),
              ),
              layer('b', result.col_mass),
            ],
            'A: 送出量 row_mass / B: 受け取り量 col_mass (1.0 = 均等)',
          )
  }
}
