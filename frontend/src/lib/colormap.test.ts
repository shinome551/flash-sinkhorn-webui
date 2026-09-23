import { describe, expect, it } from 'vitest'
import { viridis, viridisCss, viridisScaled } from './colormap.ts'

/** Rec.709 の相対輝度 (ガンマ補正なしの近似で十分。単調性の確認用)。 */
const luma = ([r, g, b]: readonly [number, number, number]) => 0.2126 * r + 0.7152 * g + 0.0722 * b

describe('viridis', () => {
  it('両端は viridis の最小・最大色', () => {
    expect(viridis(0)).toEqual([0x44, 0x01, 0x54])
    expect(viridis(1)).toEqual([0xfd, 0xe7, 0x25])
  })

  it('区間の分点はアンカーと一致し、途中は線形補間される', () => {
    expect(viridis(0.5)).toEqual([0x21, 0x90, 0x8c])
    const [r0] = viridis(0)
    const [r1] = viridis(0.125)
    expect(viridis(0.0625)[0]).toBe(Math.round((r0 + r1) / 2))
  })

  it('範囲外は端に丸め、NaN は 0 と同じ', () => {
    expect(viridis(-3)).toEqual(viridis(0))
    expect(viridis(7)).toEqual(viridis(1))
    expect(viridis(NaN)).toEqual(viridis(0))
  })

  it('明度は t に対して単調に増える', () => {
    let prev = -Infinity
    for (let k = 0; k <= 100; k++) {
      const l = luma(viridis(k / 100))
      expect(l).toBeGreaterThanOrEqual(prev)
      prev = l
    }
  })

  it('CSS 文字列に整形できる', () => {
    expect(viridisCss(0)).toBe('rgb(68, 1, 84)')
  })
})

describe('viridisScaled', () => {
  it('[lo, hi] を [0, 1] に写す', () => {
    expect(viridisScaled(2, 2, 6)).toBe(viridisCss(0))
    expect(viridisScaled(6, 2, 6)).toBe(viridisCss(1))
    expect(viridisScaled(4, 2, 6)).toBe(viridisCss(0.5))
  })

  it('定数場 (hi <= lo) は中間色', () => {
    expect(viridisScaled(3, 3, 3)).toBe(viridisCss(0.5))
    expect(viridisScaled(3, 5, 1)).toBe(viridisCss(0.5))
  })
})
