// パッチ格子の座標変換。バックエンドの規約 (SPEC 3.2, backend/app/services/patches.py) と一致させる。
//   - パッチ i = row * cols + col (unfold の出力順 = 行優先)
//   - パッチ (row, col) は画像座標 [col*stride, col*stride+size) × [row*stride, row*stride+size) の領域
//   - 代表座標はパッチ中心 (col*stride + size/2, row*stride + size/2)。ピクセル端基準
// 「画像座標」は前処理後の画像のピクセル、「表示座標」は画像座標に displayScale を掛けた CSS ピクセル。

import type { GridInfo } from '../api/types.ts'

export interface Point {
  x: number
  y: number
}

export interface Rect {
  x: number
  y: number
  width: number
  height: number
}

export type GridGeometry = Pick<GridInfo, 'rows' | 'cols' | 'patch_size' | 'stride'>

/** 画像寸法とパッチ設定から格子を求める。unfold と同じく、収まらない端は無視する。 */
export function computeGrid(
  imageWidth: number,
  imageHeight: number,
  patchSize: number,
  stride: number = patchSize,
): GridInfo {
  const count = (length: number) => (length < patchSize ? 0 : Math.floor((length - patchSize) / stride) + 1)
  return {
    rows: count(imageHeight),
    cols: count(imageWidth),
    patch_size: patchSize,
    stride,
    image_width: imageWidth,
    image_height: imageHeight,
  }
}

export function patchCount(grid: GridGeometry): number {
  return grid.rows * grid.cols
}

// ---- i ↔ (row, col) ----

export function toIndex(grid: GridGeometry, row: number, col: number): number {
  return row * grid.cols + col
}

export function fromIndex(grid: GridGeometry, i: number): { row: number; col: number } {
  return { row: Math.floor(i / grid.cols), col: i % grid.cols }
}

export function isValidIndex(grid: GridGeometry, i: number): boolean {
  return Number.isInteger(i) && i >= 0 && i < patchCount(grid)
}

// ---- (row, col) / i → 画像座標 ----

/** パッチが覆う領域 (画像座標)。 */
export function patchRect(grid: GridGeometry, i: number): Rect {
  const { row, col } = fromIndex(grid, i)
  return {
    x: col * grid.stride,
    y: row * grid.stride,
    width: grid.patch_size,
    height: grid.patch_size,
  }
}

/** パッチ中心 (画像座標)。バックエンドの `patch_centers` と同じ値。 */
export function patchCenter(grid: GridGeometry, i: number): Point {
  const { row, col } = fromIndex(grid, i)
  return {
    x: col * grid.stride + grid.patch_size / 2,
    y: row * grid.stride + grid.patch_size / 2,
  }
}

// ---- 画像座標 ↔ 表示座標 ----

export function toDisplay(p: Point, scale: number): Point {
  return { x: p.x * scale, y: p.y * scale }
}

export function fromDisplay(p: Point, scale: number): Point {
  return { x: p.x / scale, y: p.y / scale }
}

export function scaleRect(r: Rect, scale: number): Rect {
  return { x: r.x * scale, y: r.y * scale, width: r.width * scale, height: r.height * scale }
}

/** パッチが覆う領域 (表示座標)。 */
export function patchDisplayRect(grid: GridGeometry, i: number, scale: number): Rect {
  return scaleRect(patchRect(grid, i), scale)
}

/** パッチ中心 (表示座標)。接続線の端点に使う。 */
export function patchDisplayCenter(grid: GridGeometry, i: number, scale: number): Point {
  return toDisplay(patchCenter(grid, i), scale)
}

// ---- 画像座標 → パッチ (hover 検出) ----

/**
 * 画像座標の点が属するパッチの index。どのパッチにも覆われない点 (画像外、格子に収まらない端、
 * stride > size の隙間) は null。
 * stride < size でパッチが重なるときは、中心が最も近いパッチを返す (stride == size なら通常の floor と同じ)。
 */
export function patchAt(grid: GridGeometry, x: number, y: number): number | null {
  if (grid.rows <= 0 || grid.cols <= 0) return null
  const nearest = (v: number, count: number) => {
    const k = Math.round((v - grid.patch_size / 2) / grid.stride)
    return Math.min(Math.max(k, 0), count - 1)
  }
  const col = nearest(x, grid.cols)
  const row = nearest(y, grid.rows)
  const left = col * grid.stride
  const top = row * grid.stride
  const inside = x >= left && x < left + grid.patch_size && y >= top && y < top + grid.patch_size
  return inside ? toIndex(grid, row, col) : null
}

/** 表示座標 (要素内の相対位置) の点が属するパッチの index。 */
export function patchAtDisplay(grid: GridGeometry, x: number, y: number, scale: number): number | null {
  const p = fromDisplay({ x, y }, scale)
  return patchAt(grid, p.x, p.y)
}
