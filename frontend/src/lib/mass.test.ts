import { describe, expect, it } from 'vitest'
import type { MatchResponse } from '../api/types.ts'
import { isUnbalanced, massOpacity, MIN_MASS_OPACITY } from './mass.ts'

describe('massOpacity', () => {
  it('送出量 0 で下限、1 以上で 1、途中は単調', () => {
    expect(massOpacity(0)).toBeCloseTo(MIN_MASS_OPACITY)
    expect(massOpacity(1)).toBe(1)
    expect(massOpacity(3)).toBe(1)
    expect(massOpacity(0.25)).toBeCloseTo(MIN_MASS_OPACITY + (1 - MIN_MASS_OPACITY) * 0.5)
    expect(massOpacity(0.1)).toBeLessThan(massOpacity(0.5))
  })

  it('負値・NaN は 0 扱い', () => {
    expect(massOpacity(-1)).toBeCloseTo(MIN_MASS_OPACITY)
    expect(massOpacity(Number.NaN)).toBeCloseTo(MIN_MASS_OPACITY)
  })
})

describe('isUnbalanced', () => {
  const withError = (row_mass_error: number | null) => ({ stats: { row_mass_error } }) as MatchResponse
  it('row_mass_error が null なら unbalanced', () => {
    expect(isUnbalanced(withError(null))).toBe(true)
    expect(isUnbalanced(withError(0.02))).toBe(false)
  })
})
