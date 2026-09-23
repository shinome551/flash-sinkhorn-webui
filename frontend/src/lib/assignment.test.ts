import { describe, expect, it } from 'vitest'
import type { MatchResponse, PatchMatch } from '../api/types.ts'
import { viewResult } from './assignment.ts'
import { computeGrid } from './grid.ts'

const match: PatchMatch = {
  i: 0,
  confidence: 0.4,
  entropy: 1.2,
  row_mass: 0.3,
  displacement: [3, 4],
  targets: [
    { j: 2, weight: 0.4, cost: 0.1 },
    { j: 1, weight: 0.3, cost: 0.2 },
  ],
  hard: { j: 2, cost: 0.1, displacement: [16, 0] },
}

const result = {
  grid_a: computeGrid(16, 16, 16),
  grid_b: computeGrid(48, 16, 16),
  matches: [match],
  col_mass: [0.9, 1.0, 1.1],
  hard_col_mass: [0, 0, 3],
  stats: {} as MatchResponse['stats'],
  warnings: [],
} satisfies MatchResponse

describe('viewResult', () => {
  it('soft はそのまま返す', () => {
    expect(viewResult(result, 'soft')).toBe(result)
  })

  it('hard は対応先 1 件・重み 1、変位と受け取り量をハードの値に差し替える', () => {
    const view = viewResult(result, 'hard')
    expect(view.matches[0].targets).toEqual([{ j: 2, weight: 1, cost: 0.1 }])
    expect(view.matches[0].displacement).toEqual([16, 0])
    expect(view.col_mass).toEqual([0, 0, 3])
    expect(view.matches[0].row_mass).toBe(1) // 各パッチがちょうど 1 回送る
    // ソフト計画の量は残す
    expect(view.matches[0]).toMatchObject({ confidence: 0.4, entropy: 1.2 })
    expect(result.matches[0].targets).toHaveLength(2) // 元の結果は変えない
  })
})
