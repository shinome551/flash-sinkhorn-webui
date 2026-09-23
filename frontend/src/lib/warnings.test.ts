import { describe, expect, it } from 'vitest'
import { describeWarning } from './warnings.ts'

describe('describeWarning', () => {
  it('既知のコードは params から日本語の文言を作る', () => {
    expect(describeWarning({ code: 'PCA_DIM_REDUCED', message: 'x', params: { requested: 64, actual: 48 } })).toContain(
      '64 から 48',
    )
    const rowMass = describeWarning({
      code: 'ROW_MASS_ERROR',
      message: 'x',
      params: { error: 0.096, tolerance: 0.05, converged: false },
    })
    expect(rowMass).toContain('9.6%')
    expect(rowMass).toContain('反復上限')
    expect(describeWarning({ code: 'NOT_CONVERGED', message: 'x', params: { iterations: 512 } })).toContain('512 回')
  })

  it('収束していれば「反復上限」とは書かない', () => {
    const w = { code: 'ROW_MASS_ERROR', message: 'x', params: { error: 0.1, converged: true } }
    expect(describeWarning(w)).not.toContain('反復上限')
  })

  it('未知のコードや params の欠けはサーバのメッセージのまま', () => {
    expect(describeWarning({ code: 'NEW_CODE', message: 'server text', params: {} })).toBe('server text')
    expect(describeWarning({ code: 'NOT_CONVERGED', message: 'server text', params: {} })).toBe('server text')
  })
})
