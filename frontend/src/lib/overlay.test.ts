import { describe, expect, it } from 'vitest'
import type { MatchResponse, PatchMatch } from '../api/types.ts'
import { computeGrid } from './grid.ts'
import { massOpacity } from './mass.ts'
import {
  buildOverlay,
  HOVER_COLOR,
  pinColor,
  PIN_COLORS,
  weightColor,
  weightLineWidth,
  weightOpacity,
  type BoxGeometry,
} from './overlay.ts'

// A: 64×32 (2 行 × 4 列)、B: 32×32 (2 行 × 2 列)、パッチ 16px
const gridA = computeGrid(64, 32, 16)
const gridB = computeGrid(32, 32, 16)

function match(i: number, targets: [number, number][]): PatchMatch {
  return {
    i,
    confidence: targets[0]?.[1] ?? 0,
    entropy: 0,
    row_mass: 1,
    displacement: [0, 0],
    targets: targets.map(([j, weight]) => ({ j, weight, cost: 0 })),
    hard: { j: targets[0]?.[0] ?? 0, cost: 0, displacement: [0, 0] },
  }
}

const result = {
  grid_a: gridA,
  grid_b: gridB,
  matches: [
    match(0, [
      [3, 0.6],
      [0, 0.3],
    ]),
    match(1, [[1, 1]]),
    match(2, []),
    match(3, [[2, 0.5]]),
    match(4, []),
    match(5, []),
    match(6, []),
    match(7, []),
  ],
  col_mass: [],
  hard_col_mass: [],
  stats: {} as MatchResponse['stats'],
  warnings: [],
} satisfies MatchResponse

// A は (10, 20) 起点で 2 倍表示、B は (300, 5) 起点で 1 倍表示
const a: BoxGeometry = { x: 10, y: 20, scale: 2 }
const b: BoxGeometry = { x: 300, y: 5, scale: 1 }

describe('重みの見た目', () => {
  it('不透明度・太さ・色は重みに対して単調で、重み 0 でも下限が残る', () => {
    expect(weightOpacity(0)).toBeCloseTo(0.15)
    expect(weightOpacity(1)).toBeCloseTo(1)
    expect(weightLineWidth(0)).toBeCloseTo(0.75)
    expect(weightLineWidth(1)).toBeCloseTo(4)
    expect(weightOpacity(0.2)).toBeGreaterThan(weightOpacity(0.1))
    expect(weightLineWidth(0.2)).toBeGreaterThan(weightLineWidth(0.1))
    expect(weightColor(0.1)).not.toBe(weightColor(0.9))
  })

  it('範囲外・NaN は 0..1 に丸める', () => {
    expect(weightOpacity(5)).toBeCloseTo(1)
    expect(weightOpacity(-1)).toBeCloseTo(0.15)
    expect(weightOpacity(NaN)).toBeCloseTo(0.15)
  })

  it('pin の色は slot で循環する', () => {
    expect(pinColor(0)).toBe(PIN_COLORS[0])
    expect(pinColor(PIN_COLORS.length)).toBe(PIN_COLORS[0])
  })
})

describe('buildOverlay', () => {
  it('選択なしなら何も描かない', () => {
    expect(buildOverlay({ result, a, b, pins: [], active: null })).toEqual({
      rectsA: [],
      rectsB: [],
      lines: [],
    })
  })

  it('hover: A の枠、B の top-k 矩形、中心どうしを結ぶ接続線をステージ座標で作る', () => {
    const o = buildOverlay({ result, a, b, pins: [], active: 0 })
    // A のパッチ 0 は (0,0) 16×16 → 表示 2 倍でステージ (10, 20) 起点の 32×32
    expect(o.rectsA).toHaveLength(1)
    expect(o.rectsA[0]).toMatchObject({ x: 10, y: 20, width: 32, height: 32, stroke: HOVER_COLOR })
    // B の j=3 は row 1, col 1 → 画像座標 (16, 16)、1 倍表示でステージ (316, 21)
    expect(o.rectsB.map((r) => [r.x, r.y, r.width])).toEqual([
      [316, 21, 16],
      [300, 5, 16],
    ])
    // 線: A 中心 (8,8)*2+(10,20) = (26, 36) → B 中心 j=3 (24,24)+(300,5) = (324, 29)
    expect(o.lines[0]).toMatchObject({ x1: 26, y1: 36, x2: 324, y2: 29 })
    expect(o.lines).toHaveLength(2)
  })

  it('hover: 重みが大きい対応先ほど不透明で太く、色は重みで変わる', () => {
    const o = buildOverlay({ result, a, b, pins: [], active: 0 })
    const [hi, lo] = o.lines
    expect(hi.opacity).toBeGreaterThan(lo.opacity)
    expect(hi.width).toBeGreaterThan(lo.width)
    expect(hi.color).not.toBe(lo.color)
    expect(o.rectsB[0].fillOpacity).toBeGreaterThan(o.rectsB[1].fillOpacity)
  })

  it('対応先が 0 件のパッチは A の枠だけ', () => {
    const o = buildOverlay({ result, a, b, pins: [], active: 2 })
    expect(o.rectsA).toHaveLength(1)
    expect(o.rectsB).toEqual([])
    expect(o.lines).toEqual([])
  })

  it('pin: 複数を slot ごとの色で描き、線は pin の色になる', () => {
    const o = buildOverlay({
      result,
      a,
      b,
      pins: [
        { i: 1, slot: 0 },
        { i: 3, slot: 2 },
      ],
      active: null,
    })
    expect(o.rectsA.map((r) => r.stroke)).toEqual([pinColor(0), pinColor(2)])
    expect(o.lines.map((l) => l.color)).toEqual([pinColor(0), pinColor(2)])
    expect(o.rectsB.map((r) => r.stroke)).toEqual([pinColor(0), pinColor(2)])
  })

  it('pin と hover が共存し、hover が最前面 (末尾) に来る', () => {
    const o = buildOverlay({ result, a, b, pins: [{ i: 1, slot: 0 }], active: 0 })
    expect(o.rectsA.map((r) => r.key)).toEqual(['pin1-A', 'hover-A'])
    expect(o.lines).toHaveLength(3)
    expect(o.lines[o.lines.length - 1].key.startsWith('hover')).toBe(true)
  })

  it('pin 済みパッチの hover は A の枠を重ねるだけで、対応先は重複させない', () => {
    const o = buildOverlay({ result, a, b, pins: [{ i: 1, slot: 0 }], active: 1 })
    expect(o.rectsA.map((r) => r.key)).toEqual(['pin1-A', 'hover-A'])
    expect(o.rectsB).toHaveLength(1)
    expect(o.lines).toHaveLength(1)
  })

  it('存在しない index は無視する', () => {
    const o = buildOverlay({ result, a, b, pins: [{ i: 99, slot: 0 }], active: 99 })
    expect(o).toEqual({ rectsA: [], rectsB: [], lines: [] })
  })

  it('relative: 各パッチの top-1 を 1 として線の強さを決める (絶対値が小さくても濃い)', () => {
    // パッチ 3 の対応先は重み 0.5 の 1 件だけ。absolute では 0.5 相当、relative では 1 相当になる。
    const abs = buildOverlay({ result, a, b, pins: [], active: 3, weightNorm: 'absolute' })
    const rel = buildOverlay({ result, a, b, pins: [], active: 3, weightNorm: 'relative' })
    expect(rel.lines[0].opacity).toBeCloseTo(weightOpacity(1))
    expect(rel.lines[0].width).toBeCloseTo(weightLineWidth(1))
    expect(rel.lines[0].opacity).toBeGreaterThan(abs.lines[0].opacity)
    expect(abs.lines[0].opacity).toBeCloseTo(weightOpacity(0.5))
  })

  it('relative: top-1 以外は top-1 に対する比で弱くなり、順序は保たれる', () => {
    const o = buildOverlay({ result, a, b, pins: [], active: 0, weightNorm: 'relative' })
    const [first, second] = o.lines
    expect(first.opacity).toBeCloseTo(weightOpacity(1))
    expect(second.opacity).toBeCloseTo(weightOpacity(0.3 / 0.6))
    expect(first.width).toBeGreaterThan(second.width)
  })

  it('showTargets=false: hover は A の枠だけで、B の矩形と線は出さない', () => {
    const o = buildOverlay({ result, a, b, pins: [], active: 0, showTargets: false })
    expect(o.rectsA).toHaveLength(1)
    expect(o.rectsB).toEqual([])
    expect(o.lines).toEqual([])
  })

  it('unbalanced: pin の対応先は送出量で薄くし、hover は薄くしない', () => {
    const low = { ...result.matches[1], row_mass: 0.04 }
    const matches = result.matches.map((m) => (m.i === 1 ? low : m))
    const base = { a, b, weightNorm: 'relative' as const }
    const balanced = { ...result, matches, stats: { row_mass_error: 0.01 } as MatchResponse['stats'] }
    const unbalanced = { ...result, matches, stats: { row_mass_error: null } as MatchResponse['stats'] }

    const pinned = (r: MatchResponse) => buildOverlay({ ...base, result: r, pins: [{ i: 1, slot: 0 }], active: null })
    expect(pinned(balanced).lines[0].opacity).toBeCloseTo(1)
    expect(pinned(unbalanced).lines[0].opacity).toBeCloseTo(massOpacity(0.04))

    const hovered = buildOverlay({ ...base, result: unbalanced, pins: [], active: 1 })
    expect(hovered.lines[0].opacity).toBeCloseTo(1)
  })
})
