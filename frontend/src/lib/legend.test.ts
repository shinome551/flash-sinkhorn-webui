import { describe, expect, it } from 'vitest'
import { layoutLegend } from './legend.ts'

// 文字幅の近似: 1 文字 = フォントサイズの 0.6 倍
const measure = (text: string, fontSize: number) => text.length * fontSize * 0.6
const texts = { label: 'A 側パッチの変位の大きさ [px]', lo: '0.00', hi: '123.45' }
const UNIT = 64 // CAPTION 32 × SCALE 2

function overlaps(a: [number, number], b: [number, number]) {
  return a[0] < b[1] && b[0] < a[1]
}

describe('layoutLegend', () => {
  it('幅が十分なら 1 段で、ラベルと範囲が重ならない', () => {
    const l = layoutLegend(2000, UNIT, texts, measure)
    expect(l.rows).toBe(1)
    const labelEnd = l.label.x + measure(texts.label, l.fontSize)
    const loStart = l.lo.x - measure(texts.lo, l.fontSize)
    expect(labelEnd).toBeLessThan(loStart)
    expect(l.hi.x + measure(texts.hi, l.fontSize)).toBeLessThanOrEqual(2000)
  })

  it('幅が狭い (128px の画像 2 枚) と 2 段になり、どの要素も幅に収まって重ならない', () => {
    const width = (128 + 48 + 128) * 2
    const l = layoutLegend(width, UNIT, texts, measure)
    expect(l.rows).toBe(2)
    expect(l.label.x + measure(texts.label, l.fontSize)).toBeLessThanOrEqual(width)
    const lo: [number, number] = [l.lo.x - measure(texts.lo, l.fontSize), l.lo.x]
    const bar: [number, number] = [l.bar.x, l.bar.x + l.bar.width]
    const hi: [number, number] = [l.hi.x, l.hi.x + measure(texts.hi, l.fontSize)]
    expect(lo[0]).toBeGreaterThanOrEqual(0)
    expect(hi[1]).toBeLessThanOrEqual(width)
    expect(overlaps(lo, bar) || overlaps(bar, hi)).toBe(false)
    expect(l.bar.width).toBeGreaterThan(0)
  })

  it('極端に狭いと文字を縮めて収める', () => {
    const l = layoutLegend(200, UNIT, texts, measure)
    expect(l.fontSize).toBeLessThan(Math.round(UNIT * 0.4))
    expect(l.label.x + measure(texts.label, l.fontSize)).toBeLessThanOrEqual(200)
  })
})
