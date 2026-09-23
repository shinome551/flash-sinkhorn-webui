import { describe, expect, it } from 'vitest'
import {
  DEFAULT_DRAFT,
  describeRange,
  parseNumber,
  paramsOf,
  parseParams,
  parsePatch,
  sameParams,
  type ParamDraft,
} from './params.ts'

const draft = (patch: Partial<ParamDraft>): ParamDraft => ({ ...DEFAULT_DRAFT, ...patch })

describe('describeRange', () => {
  it('両端の開閉を文言に反映する', () => {
    expect(describeRange({ integer: true, min: 2, max: 64 })).toBe('2 以上 64 以下の整数')
    expect(describeRange({ min: 0, minInclusive: false, max: 10 })).toBe('0 より大きく 10 以下の数')
    expect(describeRange({ min: 0, minInclusive: false, max: 1, maxInclusive: false })).toBe('0 より大きく 1 未満の数')
    expect(describeRange({ min: 0, minInclusive: false })).toBe('0 より大きい数')
    expect(describeRange({ integer: true, min: 1 })).toBe('1 以上の整数')
  })
})

describe('parseNumber', () => {
  it('指数表記と前後の空白を受け付ける', () => {
    expect(parseNumber(' 1e-3 ', { min: 0, minInclusive: false })).toEqual({ ok: true, value: 0.001 })
    expect(parseNumber('.5', {})).toEqual({ ok: true, value: 0.5 })
  })

  it('空欄は optional のときだけ null', () => {
    expect(parseNumber('', { optional: true })).toEqual({ ok: true, value: null })
    expect(parseNumber('  ', {})).toMatchObject({ ok: false })
  })

  it('数値でない文字列・途中の入力・16 進を拒否する', () => {
    for (const text of ['abc', '1e', '1e-', '0x10', '1,5', '--1', 'Infinity', 'NaN']) {
      expect(parseNumber(text, {}), text).toMatchObject({ ok: false })
    }
  })

  it('整数指定では小数を拒否する', () => {
    expect(parseNumber('16.5', { integer: true })).toMatchObject({ ok: false })
    expect(parseNumber('16.0', { integer: true })).toEqual({ ok: true, value: 16 })
  })

  it('範囲の端: 含む端は通り、含まない端は拒否する', () => {
    const spec = { min: 0, minInclusive: false, max: 1, maxInclusive: false }
    expect(parseNumber('0', spec)).toMatchObject({ ok: false })
    expect(parseNumber('1', spec)).toMatchObject({ ok: false })
    expect(parseNumber('0.999', spec)).toMatchObject({ ok: true })
    expect(parseNumber('2', { min: 2, max: 64 })).toMatchObject({ ok: true })
    expect(parseNumber('64', { min: 2, max: 64 })).toMatchObject({ ok: true })
    expect(parseNumber('65', { min: 2, max: 64 })).toMatchObject({ ok: false })
  })
})

describe('parseParams', () => {
  it('既定のドラフトは、バックエンドの既定値と同じリクエストになる', () => {
    const parsed = parseParams(DEFAULT_DRAFT)
    expect(parsed).toEqual({
      ok: true,
      params: {
        patch: { size: 16, stride: null },
        feature: { type: 'pca', pca_dim: 64, normalize: 'zscore', position_weight: 0 },
        ot: {
          blur: 0.05,
          scaling: 0.5,
          half_cost: false,
          reach_x: null,
          reach_y: null,
          threshold: 0.001,
          inner_iterations: 10,
          backend: null,
        },
        output: { top_k: 3, min_weight: 0.0001, compute_divergence: true },
      },
    })
  })

  it('入力した値をそのまま反映する', () => {
    const parsed = parseParams(
      draft({
        patchSize: '8',
        stride: '4',
        blur: '0.2',
        reachX: '0.5',
        threshold: '',
        backend: 'dense',
        halfCost: true,
      }),
    )
    if (!parsed.ok) throw new Error('should parse')
    expect(parsed.params.patch).toEqual({ size: 8, stride: 4 })
    expect(parsed.params.ot).toMatchObject({
      blur: 0.2,
      reach_x: 0.5,
      reach_y: null,
      threshold: null,
      backend: 'dense',
      half_cost: true,
    })
  })

  it('不正な項目をすべて列挙する', () => {
    const parsed = parseParams(draft({ patchSize: '1', blur: '0', scaling: '1', topK: '11', reachY: '-1' }))
    if (parsed.ok) throw new Error('should fail')
    expect(Object.keys(parsed.errors).sort()).toEqual(['blur', 'patchSize', 'reachY', 'scaling', 'topK'])
  })
})

describe('sameParams', () => {
  it('内容が同じなら true、違えば false、null は null とだけ一致する', () => {
    const a = parseParams(DEFAULT_DRAFT)
    const b = parseParams(DEFAULT_DRAFT)
    const c = parseParams(draft({ blur: '0.1' }))
    if (!a.ok || !b.ok || !c.ok) throw new Error('should parse')
    expect(sameParams(a.params, b.params)).toBe(true)
    expect(sameParams(a.params, c.params)).toBe(false)
    expect(sameParams(null, null)).toBe(true)
    expect(sameParams(a.params, null)).toBe(false)
  })
})

describe('parsePatch', () => {
  it('stride が空欄なら size と同じ', () => {
    expect(parsePatch(draft({ patchSize: '8' }))).toEqual({ size: 8, stride: 8 })
    expect(parsePatch(draft({ patchSize: '8', stride: '4' }))).toEqual({ size: 8, stride: 4 })
  })

  it('パッチ設定が不正なら null。他の項目が不正でも影響しない', () => {
    expect(parsePatch(draft({ patchSize: '1' }))).toBeNull()
    expect(parsePatch(draft({ stride: '0' }))).toBeNull()
    expect(parsePatch(draft({ blur: 'abc' }))).toEqual({ size: 16, stride: 16 })
  })
})

describe('paramsOf', () => {
  it('parseParams の結果を含むリクエストから、同じパラメータを取り出す', () => {
    const parsed = parseParams(DEFAULT_DRAFT)
    if (!parsed.ok) throw new Error('should parse')
    const request = { image_a: 'a', image_b: 'b', ...parsed.params }
    expect(sameParams(paramsOf(request), parsed.params)).toBe(true)
    expect(paramsOf({ image_a: 'a', image_b: 'b' })).toBeNull()
  })
})
