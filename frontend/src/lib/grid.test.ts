import { describe, expect, it } from 'vitest'
import {
  computeGrid,
  fromDisplay,
  fromIndex,
  isValidIndex,
  patchAt,
  patchAtDisplay,
  patchCenter,
  patchCount,
  patchDisplayCenter,
  patchDisplayRect,
  patchRect,
  toDisplay,
  toIndex,
} from './grid.ts'

// 期待値のうち computeGrid / patchCenter は backend の compute_grid / patch_centers の実出力
// (backend/app/services/patches.py) から取った値。規約がずれたらここで検出する。
describe('computeGrid', () => {
  it('SPEC の例: 512x384 / 16 → 24 行 32 列', () => {
    expect(computeGrid(512, 384, 16)).toEqual({
      rows: 24,
      cols: 32,
      patch_size: 16,
      stride: 16,
      image_width: 512,
      image_height: 384,
    })
  })

  it('stride < size (重なりあり): 50x40 / 16 / 8 → 4 行 5 列', () => {
    const g = computeGrid(50, 40, 16, 8)
    expect([g.rows, g.cols]).toEqual([4, 5])
  })

  it('stride > size (隙間あり): 100x60 / 16 / 24 → 2 行 4 列', () => {
    const g = computeGrid(100, 60, 16, 24)
    expect([g.rows, g.cols]).toEqual([2, 4])
  })

  it('格子に収まらない端は無視する / パッチより小さい辺は 0 個', () => {
    expect(computeGrid(50, 40, 16).cols).toBe(3)
    expect(computeGrid(15, 40, 16).cols).toBe(0)
  })
})

describe('i ↔ (row, col)', () => {
  const g = computeGrid(512, 384, 16)

  it('行優先で往復できる', () => {
    expect(toIndex(g, 0, 0)).toBe(0)
    expect(toIndex(g, 1, 0)).toBe(32)
    expect(toIndex(g, 23, 31)).toBe(767)
    for (const i of [0, 1, 31, 32, 100, 767]) {
      const { row, col } = fromIndex(g, i)
      expect(toIndex(g, row, col)).toBe(i)
    }
    expect(fromIndex(g, 33)).toEqual({ row: 1, col: 1 })
  })

  it('index の妥当性', () => {
    expect(patchCount(g)).toBe(768)
    expect(isValidIndex(g, 0)).toBe(true)
    expect(isValidIndex(g, 767)).toBe(true)
    expect(isValidIndex(g, 768)).toBe(false)
    expect(isValidIndex(g, -1)).toBe(false)
    expect(isValidIndex(g, 1.5)).toBe(false)
  })
})

describe('patchCenter / patchRect', () => {
  it('backend の patch_centers と一致 (50x40 / 16 / 8)', () => {
    const g = computeGrid(50, 40, 16, 8)
    expect(patchCenter(g, 0)).toEqual({ x: 8, y: 8 })
    expect(patchCenter(g, 1)).toEqual({ x: 16, y: 8 })
    expect(patchCenter(g, g.cols)).toEqual({ x: 8, y: 16 })
    expect(patchCenter(g, patchCount(g) - 1)).toEqual({ x: 40, y: 32 })
  })

  it('backend の patch_centers と一致 (100x60 / 16 / 24)', () => {
    const g = computeGrid(100, 60, 16, 24)
    const centers = Array.from({ length: patchCount(g) }, (_, i) => patchCenter(g, i))
    expect(centers.map((c) => [c.x, c.y])).toEqual([
      [8, 8],
      [32, 8],
      [56, 8],
      [80, 8],
      [8, 32],
      [32, 32],
      [56, 32],
      [80, 32],
    ])
  })

  it('領域はパッチ中心を中心に持つ', () => {
    const g = computeGrid(50, 40, 16, 8)
    for (let i = 0; i < patchCount(g); i++) {
      const r = patchRect(g, i)
      const c = patchCenter(g, i)
      expect(r.x + r.width / 2).toBe(c.x)
      expect(r.y + r.height / 2).toBe(c.y)
      expect(r.width).toBe(16)
    }
    expect(patchRect(g, 6)).toEqual({ x: 8, y: 8, width: 16, height: 16 })
  })
})

describe('画像座標 ↔ 表示座標', () => {
  it('往復で元に戻る', () => {
    const p = { x: 12.5, y: 33 }
    for (const s of [0.5, 1, 1.25, 2]) {
      const q = fromDisplay(toDisplay(p, s), s)
      expect(q.x).toBeCloseTo(p.x)
      expect(q.y).toBeCloseTo(p.y)
    }
  })

  it('表示スケールが領域と中心に反映される', () => {
    const g = computeGrid(64, 48, 16)
    expect(patchDisplayRect(g, 5, 1.5)).toEqual({ x: 24, y: 24, width: 24, height: 24 })
    expect(patchDisplayCenter(g, 5, 1.5)).toEqual({ x: 36, y: 36 })
  })
})

describe('patchAt (hover 検出)', () => {
  it('stride == size: 各パッチの領域内はそのパッチ、境界は次のパッチ', () => {
    const g = computeGrid(64, 48, 16) // 3 行 4 列
    expect(patchAt(g, 0, 0)).toBe(0)
    expect(patchAt(g, 15.99, 15.99)).toBe(0)
    expect(patchAt(g, 16, 0)).toBe(1)
    expect(patchAt(g, 0, 16)).toBe(4)
    expect(patchAt(g, 63.9, 47.9)).toBe(11)
  })

  it('全パッチで、中心と領域の四隅がそのパッチに解決される', () => {
    const g = computeGrid(64, 48, 16)
    for (let i = 0; i < patchCount(g); i++) {
      const c = patchCenter(g, i)
      expect(patchAt(g, c.x, c.y)).toBe(i)
      const r = patchRect(g, i)
      expect(patchAt(g, r.x, r.y)).toBe(i)
      expect(patchAt(g, r.x + r.width - 0.01, r.y + r.height - 0.01)).toBe(i)
    }
  })

  it('画像外・格子に収まらない端は null', () => {
    const g = computeGrid(50, 40, 16) // 有効領域は 48x32
    expect(patchAt(g, -0.1, 5)).toBeNull()
    expect(patchAt(g, 5, -0.1)).toBeNull()
    expect(patchAt(g, 48, 5)).toBeNull()
    expect(patchAt(g, 5, 32)).toBeNull()
    expect(patchAt(g, 47.9, 31.9)).toBe(5)
  })

  it('stride < size (重なり): 中心が最も近いパッチを返し、そのパッチが点を含む', () => {
    const g = computeGrid(50, 40, 16, 8) // 4 行 5 列
    expect(patchAt(g, 8, 8)).toBe(0)
    expect(patchAt(g, 15, 8)).toBe(1) // x=15 は col0 (中心 8) と col1 (中心 16) の両方に含まれ、col1 が近い
    expect(patchAt(g, 11, 8)).toBe(0) // 中心 8 に近い
    for (let y = 0; y < 32; y += 0.7) {
      for (let x = 0; x < 48; x += 0.7) {
        const i = patchAt(g, x, y)
        expect(i).not.toBeNull()
        const r = patchRect(g, i!)
        expect(x >= r.x && x < r.x + r.width && y >= r.y && y < r.y + r.height).toBe(true)
      }
    }
  })

  it('stride > size (隙間): 隙間は null', () => {
    const g = computeGrid(100, 60, 16, 24)
    expect(patchAt(g, 8, 8)).toBe(0)
    expect(patchAt(g, 20, 8)).toBeNull() // col0 は [0,16)、col1 は [24,40)
    expect(patchAt(g, 24, 8)).toBe(1)
    expect(patchAt(g, 39.9, 8)).toBe(1)
    expect(patchAt(g, 40, 8)).toBeNull()
  })

  it('空の格子は常に null', () => {
    expect(patchAt(computeGrid(15, 40, 16), 5, 5)).toBeNull()
  })

  it('表示座標版はスケールを外してから判定する', () => {
    const g = computeGrid(64, 48, 16)
    expect(patchAtDisplay(g, 30, 30, 2)).toBe(0) // 画像座標 (15, 15)
    expect(patchAtDisplay(g, 34, 30, 2)).toBe(1) // 画像座標 (17, 15)
    expect(patchAtDisplay(g, 200, 30, 2)).toBeNull()
  })
})
