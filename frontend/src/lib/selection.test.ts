import { describe, expect, it } from 'vitest'
import { computeGrid } from './grid.ts'
import { movePatch, togglePin } from './selection.ts'

// 4 行 × 6 列
const grid = computeGrid(96, 64, 16)

describe('togglePin', () => {
  it('未固定なら追加し、固定済みなら解除する', () => {
    const one = togglePin([], 5)
    expect(one).toEqual([{ i: 5, slot: 0 }])
    expect(togglePin(one, 5)).toEqual([])
  })

  it('slot は使われていない最小の番号で、解除しても他の pin の slot は変わらない', () => {
    let pins = togglePin([], 1)
    pins = togglePin(pins, 2)
    pins = togglePin(pins, 3)
    expect(pins.map((p) => p.slot)).toEqual([0, 1, 2])
    pins = togglePin(pins, 2) // slot 1 を解放
    expect(pins.map((p) => [p.i, p.slot])).toEqual([
      [1, 0],
      [3, 2],
    ])
    pins = togglePin(pins, 9)
    expect(pins.find((p) => p.i === 9)?.slot).toBe(1)
  })

  it('入力の配列を書き換えない', () => {
    const pins = [{ i: 1, slot: 0 }]
    togglePin(pins, 2)
    expect(pins).toEqual([{ i: 1, slot: 0 }])
  })
})

describe('movePatch', () => {
  it('未選択からは格子の中央から始める', () => {
    expect(movePatch(grid, null, 'ArrowRight')).toBe(2 * 6 + 3)
  })

  it('上下左右に 1 つ動く', () => {
    const from = 1 * 6 + 2
    expect(movePatch(grid, from, 'ArrowUp')).toBe(2)
    expect(movePatch(grid, from, 'ArrowDown')).toBe(2 * 6 + 2)
    expect(movePatch(grid, from, 'ArrowLeft')).toBe(1 * 6 + 1)
    expect(movePatch(grid, from, 'ArrowRight')).toBe(1 * 6 + 3)
  })

  it('端では留まる (行をまたがない)', () => {
    expect(movePatch(grid, 0, 'ArrowUp')).toBe(0)
    expect(movePatch(grid, 0, 'ArrowLeft')).toBe(0)
    expect(movePatch(grid, 6, 'ArrowLeft')).toBe(6)
    expect(movePatch(grid, 5, 'ArrowRight')).toBe(5)
    expect(movePatch(grid, 3 * 6 + 5, 'ArrowDown')).toBe(3 * 6 + 5)
  })

  it('step で複数個動き、端で丸める', () => {
    expect(movePatch(grid, 0, 'ArrowRight', 4)).toBe(4)
    expect(movePatch(grid, 0, 'ArrowRight', 99)).toBe(5)
    expect(movePatch(grid, 0, 'ArrowDown', 99)).toBe(3 * 6)
  })

  it('範囲外の from は未選択として扱い、空の格子は null', () => {
    expect(movePatch(grid, 999, 'ArrowUp')).toBe(2 * 6 + 3)
    expect(movePatch(computeGrid(8, 8, 16), null, 'ArrowUp')).toBeNull()
  })
})
