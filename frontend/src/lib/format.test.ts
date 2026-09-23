import { describe, expect, it } from 'vitest'
import { formatDelta, formatNumber, formatRange } from './format.ts'

describe('formatNumber', () => {
  it('有効数字 3 桁で、末尾の 0 は落とす', () => {
    expect(formatNumber(0.012345)).toBe('0.0123')
    expect(formatNumber(1.5)).toBe('1.5')
    expect(formatNumber(1234.56)).toBe('1230')
    expect(formatNumber(1)).toBe('1')
  })

  it('極端な値は指数表記、0 と非有限値はそのまま', () => {
    expect(formatNumber(0.0025)).toBe('0.0025')
    expect(formatNumber(0.00012345)).toBe('1.23e-4')
    expect(formatNumber(123456)).toBe('1.23e+5')
    expect(formatNumber(0)).toBe('0')
    expect(formatNumber(Number.NaN)).toBe('NaN')
  })
})

describe('formatDelta', () => {
  it('符号をつける', () => {
    expect(formatDelta(0.5)).toBe('+0.5')
    expect(formatDelta(-0.25)).toBe('−0.25')
    expect(formatDelta(0)).toBe('±0')
  })
})

describe('formatRange', () => {
  it('通常の範囲は 3 桁', () => {
    expect(formatRange(0.0123456, 0.98765)).toEqual(['0.0123', '0.988'])
  })

  it('狭い範囲は、幅が読み取れる桁数まで増やす', () => {
    expect(formatRange(0.99981, 1.0002)).toEqual(['0.99981', '1.0002'])
  })

  it('幅が 0 なら 3 桁のまま', () => {
    expect(formatRange(2, 2)).toEqual(['2', '2'])
  })
})
