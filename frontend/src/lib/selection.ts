// A 側パッチの選択状態 (pin と矢印キー移動) の純粋なロジック。

import { fromIndex, patchCount, toIndex, type GridGeometry } from './grid.ts'

/** 固定したパッチ。`slot` は色の割り当て番号で、解除しても他の pin の色は変わらない。 */
export interface Pin {
  i: number
  slot: number
}

/** pin 済みなら解除、未固定なら追加する。追加時の slot は使われていない最小の番号。 */
export function togglePin(pins: readonly Pin[], i: number): Pin[] {
  if (pins.some((p) => p.i === i)) return pins.filter((p) => p.i !== i)
  const used = new Set(pins.map((p) => p.slot))
  let slot = 0
  while (used.has(slot)) slot++
  return [...pins, { i, slot }]
}

export type ArrowKey = 'ArrowUp' | 'ArrowDown' | 'ArrowLeft' | 'ArrowRight'

export function isArrowKey(key: string): key is ArrowKey {
  return key === 'ArrowUp' || key === 'ArrowDown' || key === 'ArrowLeft' || key === 'ArrowRight'
}

/**
 * 矢印キーによる移動先。`from` が null のときは格子の中央から始める (移動はしない)。
 * 端では留まる。格子が空なら null。
 */
export function movePatch(grid: GridGeometry, from: number | null, key: ArrowKey, step = 1): number | null {
  if (patchCount(grid) <= 0) return null
  if (from === null || from < 0 || from >= patchCount(grid)) {
    return toIndex(grid, Math.floor(grid.rows / 2), Math.floor(grid.cols / 2))
  }
  const { row, col } = fromIndex(grid, from)
  const clamp = (v: number, count: number) => Math.min(Math.max(v, 0), count - 1)
  switch (key) {
    case 'ArrowUp':
      return toIndex(grid, clamp(row - step, grid.rows), col)
    case 'ArrowDown':
      return toIndex(grid, clamp(row + step, grid.rows), col)
    case 'ArrowLeft':
      return toIndex(grid, row, clamp(col - step, grid.cols))
    case 'ArrowRight':
      return toIndex(grid, row, clamp(col + step, grid.cols))
  }
}
