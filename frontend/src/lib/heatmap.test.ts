import { describe, expect, it } from 'vitest'
import type { MatchResponse, PatchMatch } from '../api/types.ts'
import { buildHeatmap, colorRange, COVERAGE_MIN_HALF_SPAN, heatmapLayer, quantile } from './heatmap.ts'
import { computeGrid } from './grid.ts'

describe('quantile / colorRange', () => {
  it('昇順配列の分位点を返し、空なら 0', () => {
    expect(quantile([], 0.5)).toBe(0)
    expect(quantile([1, 2, 3, 4, 5], 0)).toBe(1)
    expect(quantile([1, 2, 3, 4, 5], 1)).toBe(5)
    expect(quantile([1, 2, 3, 4, 5], 0.5)).toBe(3)
  })

  it('突出した 1 点は色範囲から外れる', () => {
    const values = [...Array.from({ length: 199 }, (_, k) => 1 + k / 1000), 1000]
    const [lo, hi] = colorRange(values)
    expect(lo).toBeCloseTo(1, 1)
    expect(hi).toBeLessThan(2)
  })

  it('定数場は lo == hi、非有限値は無視する', () => {
    expect(colorRange([2, 2, 2])).toEqual([2, 2])
    expect(colorRange([1, Number.NaN, 3, Number.POSITIVE_INFINITY])).toEqual([1, 3])
  })
})

function match(i: number, confidence: number, entropy: number, row_mass = 1): PatchMatch {
  return {
    i,
    confidence,
    entropy,
    row_mass,
    displacement: [0, 0],
    targets: [],
    hard: { j: 0, cost: 0, displacement: [0, 0] },
  }
}

describe('buildHeatmap', () => {
  const result = {
    grid_a: computeGrid(32, 16, 16),
    grid_b: computeGrid(16, 32, 16),
    matches: [match(0, 0.1, 3, 0.2), match(1, 0.9, 1, 1.8)],
    col_mass: [1.5, 0.5],
    hard_col_mass: [2, 0],
    stats: {} as MatchResponse['stats'],
    warnings: [],
  } satisfies MatchResponse

  it('confidence / entropy は A 側、coverage は A (row_mass) と B (col_mass) の両側を塗る', () => {
    const conf = buildHeatmap(result, 'confidence')
    expect(conf.layers).toEqual([{ side: 'a', values: [0.1, 0.9], grid: result.grid_a }])
    const ent = buildHeatmap(result, 'entropy')
    expect(ent.layers).toMatchObject([{ side: 'a', values: [3, 1] }])
    const cov = buildHeatmap(result, 'coverage')
    expect(heatmapLayer(cov, 'a')).toEqual({ side: 'a', values: [0.2, 1.8], grid: result.grid_a })
    expect(heatmapLayer(cov, 'b')).toEqual({ side: 'b', values: [1.5, 0.5], grid: result.grid_b })
  })

  it('coverage: ハード割当では B 側だけ', () => {
    const cov = buildHeatmap(result, 'coverage', 'hard')
    expect(cov.layers.map((l) => l.side)).toEqual(['b'])
    expect(heatmapLayer(cov, 'a')).toBeUndefined()
    expect(heatmapLayer(null, 'b')).toBeUndefined()
  })

  it('coverage: 色範囲は A・B の値をまとめて決める (両側で同じ色が同じ値)', () => {
    const { lo, hi } = buildHeatmap({ ...result, col_mass: [1, 1] }, 'coverage')
    expect(lo).toBeCloseTo(0.2)
    expect(hi).toBeCloseTo(1.8)
  })

  it('coverage の凡例はハード割当かどうかで変わる', () => {
    expect(buildHeatmap(result, 'coverage').label).toContain('row_mass')
    expect(buildHeatmap(result, 'coverage', 'hard').label).toContain('ハード割当')
  })

  it('色範囲は値の範囲内に収まる', () => {
    const { lo, hi } = buildHeatmap(result, 'confidence')
    expect(lo).toBeGreaterThanOrEqual(0.1)
    expect(hi).toBeLessThanOrEqual(0.9)
  })

  it('coverage: 1.0 の近傍に収まる (balanced) ときは、最低幅 ±COVERAGE_MIN_HALF_SPAN を確保する', () => {
    const flat = { ...result, matches: [match(0, 0.1, 3), match(1, 0.9, 1)], col_mass: [1.00002, 0.99998] }
    const { lo, hi } = buildHeatmap(flat, 'coverage')
    expect(lo).toBeCloseTo(1 - COVERAGE_MIN_HALF_SPAN)
    expect(hi).toBeCloseTo(1 + COVERAGE_MIN_HALF_SPAN)
  })

  it('coverage: 偏りが最低幅を超えるときはデータの範囲に広がる', () => {
    const { lo, hi } = buildHeatmap({ ...result, col_mass: [0, 1, 1, 3] }, 'coverage')
    expect(lo).toBeLessThan(0.9)
    expect(hi).toBeGreaterThan(1.1)
  })
})
