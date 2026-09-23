import { describe, expect, it } from 'vitest'
import type { MatchResponse, PatchMatch } from '../api/types.ts'
import { arrowHead, buildFlow, confidenceOpacity, MIN_ARROW_LENGTH, MIN_FLOW_OPACITY } from './flow.ts'
import { computeGrid } from './grid.ts'
import { massOpacity } from './mass.ts'

// A: 64×48 (3 行 × 4 列)、パッチ 16px。全パッチの変位は指定値。
const gridA = computeGrid(64, 48, 16)

function resultWith(
  displacement: (i: number) => [number, number],
  confidence: (i: number) => number = () => 0.5,
  rowMass: ((i: number) => number) | null = null,
): MatchResponse {
  const n = gridA.rows * gridA.cols
  const matches: PatchMatch[] = Array.from({ length: n }, (_, i) => ({
    i,
    confidence: confidence(i),
    entropy: 1,
    row_mass: rowMass ? rowMass(i) : 1,
    displacement: displacement(i),
    targets: [],
    hard: { j: 0, cost: 0, displacement: [0, 0] },
  }))
  return {
    grid_a: gridA,
    grid_b: gridA,
    matches,
    col_mass: [],
    hard_col_mass: [],
    // rowMass を渡したら unbalanced の結果 (row_mass_error = null) とする
    stats: { row_mass_error: rowMass ? null : 0 } as MatchResponse['stats'],
    warnings: [],
  }
}

const a = { x: 10, y: 20, scale: 2 }

describe('buildFlow', () => {
  it('矢印は A のパッチ中心から、変位 × 表示倍率だけ伸びる (ステージ座標)', () => {
    const flow = buildFlow(
      resultWith(() => [8, -4]),
      a,
      1,
    )
    expect(flow.arrows).toHaveLength(12)
    // パッチ 0 の中心 (8, 8) → 表示 (16, 16) + (10, 20) = (26, 36)。変位 (8, -4) は 2 倍で (16, -8)
    expect(flow.arrows[0]).toMatchObject({ x1: 26, y1: 36, x2: 42, y2: 28 })
  })

  it('間引き: step 個おきの行・列だけを出す', () => {
    const flow = buildFlow(
      resultWith(() => [8, 0]),
      a,
      2,
    )
    // 行 0, 2 × 列 0, 2 → i = 0, 2, 8, 10
    expect(flow.arrows.map((f) => f.key)).toEqual(['flow0', 'flow2', 'flow8', 'flow10'])
  })

  it('色範囲は間引きに依存しない', () => {
    const result = resultWith((i) => [i, 0])
    const dense = buildFlow(result, a, 1)
    const sparse = buildFlow(result, a, 3)
    expect([sparse.lo, sparse.hi]).toEqual([dense.lo, dense.hi])
  })

  it('大きい変位ほど別の色になる', () => {
    const flow = buildFlow(
      resultWith((i) => [i * 10, 0]),
      a,
      1,
    )
    expect(flow.arrows[1].color).not.toBe(flow.arrows[11].color)
  })

  it('短すぎる矢印は矢じりなし (点)、十分長ければ矢じりあり', () => {
    const zero = buildFlow(
      resultWith(() => [0, 0]),
      a,
      1,
    )
    expect(zero.arrows.every((f) => f.head === null)).toBe(true)
    const long = buildFlow(
      resultWith(() => [MIN_ARROW_LENGTH, 0]),
      a,
      1,
    )
    expect(long.arrows.every((f) => f.head !== null)).toBe(true)
  })

  it('クリップ範囲は A の画像領域', () => {
    expect(
      buildFlow(
        resultWith(() => [0, 0]),
        a,
        1,
      ).clip,
    ).toEqual({ x: 10, y: 20, width: 128, height: 96 })
  })

  it('step が 1 未満・小数でも 1 以上の整数に丸める', () => {
    expect(
      buildFlow(
        resultWith(() => [1, 0]),
        a,
        0,
      ).arrows,
    ).toHaveLength(12)
    expect(
      buildFlow(
        resultWith(() => [1, 0]),
        a,
        1.6,
      ).arrows,
    ).toHaveLength(4)
  })
})

describe('arrowHead', () => {
  it('先端は終点で、底辺は進行方向に垂直', () => {
    const [tip, left, right] = arrowHead(0, 0, 10, 0, 4)
    expect(tip).toEqual({ x: 10, y: 0 })
    expect(left.x).toBeCloseTo(6)
    expect(right.x).toBeCloseTo(6)
    expect(Math.abs(left.y)).toBeCloseTo(2)
    expect(left.y + right.y).toBeCloseTo(0)
  })
})

describe('confidenceOpacity', () => {
  it('確信度の範囲を MIN_FLOW_OPACITY..1 に写す (範囲外は飽和)', () => {
    const opacity = confidenceOpacity([0.1, 0.2, 0.3, 0.4, 0.5])
    expect(opacity(0.1)).toBeCloseTo(MIN_FLOW_OPACITY)
    expect(opacity(0.5)).toBeCloseTo(1)
    expect(opacity(0.3)).toBeCloseTo((MIN_FLOW_OPACITY + 1) / 2)
    expect(opacity(-1)).toBeCloseTo(MIN_FLOW_OPACITY)
    expect(opacity(2)).toBeCloseTo(1)
  })

  it('確信度がすべて同じなら薄くしない', () => {
    expect(confidenceOpacity([0.3, 0.3, 0.3])(0.3)).toBe(1)
  })

  it('buildFlow: fadeByConfidence のときだけ不透明度が確信度に連動する', () => {
    const result = resultWith(
      () => [8, 0],
      (i) => i / 11,
    )
    expect(buildFlow(result, a, 1).arrows.every((f) => f.opacity === 1)).toBe(true)
    const faded = buildFlow(result, a, 1, true).arrows
    expect(faded[0].opacity).toBeCloseTo(MIN_FLOW_OPACITY)
    expect(faded[11].opacity).toBeCloseTo(1)
    expect(faded[3].opacity).toBeLessThan(faded[8].opacity)
  })

  it('buildFlow: unbalanced では送出量の小さい矢印を薄くし、確信度とは小さい方を取る', () => {
    const result = resultWith(
      () => [8, 0],
      (i) => i / 11,
      (i) => (i === 0 ? 0 : i === 1 ? 0.25 : 1),
    )
    const plain = buildFlow(result, a, 1).arrows
    expect(plain[0].opacity).toBeCloseTo(MIN_FLOW_OPACITY)
    expect(plain[1].opacity).toBeCloseTo(massOpacity(0.25))
    expect(plain[2].opacity).toBe(1)
    const faded = buildFlow(result, a, 1, true).arrows
    const byConfidence = confidenceOpacity(result.matches.map((m) => m.confidence))(1 / 11)
    expect(byConfidence).toBeLessThan(massOpacity(0.25))
    expect(faded[1].opacity).toBeCloseTo(byConfidence) // 確信度で薄くした方が小さい
    expect(faded[11].opacity).toBeCloseTo(1)
  })

  it('buildFlow: balanced (row_mass_error あり) では row_mass がずれていても薄くしない', () => {
    const result = resultWith(() => [8, 0])
    result.matches[0].row_mass = 0.3 // 未収束の balanced ではありうる
    expect(buildFlow(result, a, 1).arrows[0].opacity).toBe(1)
  })
})
