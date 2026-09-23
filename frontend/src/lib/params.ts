// パラメータ入力欄の状態 (ParamDraft) と、`POST /api/match` のリクエストへの変換・検証。
// 範囲と既定値はバックエンドの Pydantic スキーマ (backend/app/schemas/match.py) に合わせる。食い違ったらそちらが正。
// 数値は画面のプリセット (PARAM_CONTROLS) から選ぶ。空欄 = null を表せるよう、値は文字列のまま持つ。

import type { BackendChoice, FeatureType, MatchRequest, NormalizeMode } from '../api/types.ts'

export interface ParamDraft {
  patchSize: string
  /** 空欄 = patch.size と同じ */
  stride: string
  featureType: FeatureType
  pcaDim: string
  normalize: NormalizeMode
  positionWeight: string
  blur: string
  scaling: string
  halfCost: boolean
  /** 空欄 = null (その側は質量保存を緩めない) */
  reachX: string
  reachY: string
  /** 空欄 = null (打ち切りなし) */
  threshold: string
  innerIterations: string
  /** '' = サーバ設定に任せる (リクエストでは null) */
  backend: '' | BackendChoice
  topK: string
  minWeight: string
  computeDivergence: boolean
}

export const DEFAULT_DRAFT: ParamDraft = {
  patchSize: '16',
  stride: '',
  featureType: 'pca',
  pcaDim: '64',
  normalize: 'zscore',
  positionWeight: '0',
  blur: '0.05',
  scaling: '0.5',
  halfCost: false,
  reachX: '',
  reachY: '',
  threshold: '0.001',
  innerIterations: '10',
  backend: '',
  topK: '3',
  minWeight: '0.0001',
  computeDivergence: true,
}

export interface NumberSpec {
  integer?: boolean
  min?: number
  /** 既定 true。false なら min を含まない (min < x) */
  minInclusive?: boolean
  max?: number
  /** 既定 true。false なら max を含まない (x < max) */
  maxInclusive?: boolean
  /** true なら空欄を許す (値は null) */
  optional?: boolean
}

export type NumericKey =
  | 'patchSize'
  | 'stride'
  | 'pcaDim'
  | 'positionWeight'
  | 'blur'
  | 'scaling'
  | 'reachX'
  | 'reachY'
  | 'threshold'
  | 'innerIterations'
  | 'topK'
  | 'minWeight'

export const NUMBER_SPECS: Record<NumericKey, NumberSpec> = {
  patchSize: { integer: true, min: 2, max: 64 },
  stride: { integer: true, min: 1, optional: true },
  pcaDim: { integer: true, min: 1, max: 256 },
  positionWeight: { min: 0, max: 100 },
  blur: { min: 0, minInclusive: false, max: 10 },
  scaling: { min: 0, minInclusive: false, max: 1, maxInclusive: false },
  reachX: { min: 0, minInclusive: false, optional: true },
  reachY: { min: 0, minInclusive: false, optional: true },
  threshold: { min: 0, minInclusive: false, optional: true },
  innerIterations: { integer: true, min: 1, max: 1000 },
  topK: { integer: true, min: 1, max: 10 },
  minWeight: { min: 0, max: 1, maxInclusive: false },
}

export interface Choice {
  value: string
  label: string
}

/** 数値項目の入力部品と、選べる値 (昇順。空欄 = null は意味の上で並ぶ位置に置く)。 */
export interface ParamControl {
  kind: 'slider' | 'select'
  choices: Choice[]
}

const ints = (from: number, to: number) => Array.from({ length: to - from + 1 }, (_, i) => String(from + i))

/** 0.01 未満の正の値は 1e-4 のような指数表記にする。 */
const numberLabel = (v: string) => {
  const x = Number(v)
  return x > 0 && x < 0.01 ? x.toExponential() : v
}

const choices = (values: string[], blank?: string): Choice[] =>
  values.map((value) => ({ value, label: value === '' ? (blank ?? '') : numberLabel(value) }))

const REACH = ['0.05', '0.1', '0.2', '0.3', '0.5', '0.7', '1', '2', '5', '10', '']

/**
 * 各数値項目の選択肢。値は NUMBER_SPECS の範囲内で、DEFAULT_DRAFT の値を含む (テストで確認)。
 * patch size / stride はサーバの hint (「size 19 以上」など) をそのまま選べるよう整数をすべて並べる。
 * reach は大きいほど balanced に近いので、balanced (空欄) を右端に置く。
 */
export const PARAM_CONTROLS: Record<NumericKey, ParamControl> = {
  patchSize: { kind: 'slider', choices: choices(ints(2, 64)) },
  stride: { kind: 'slider', choices: choices(['', ...ints(1, 64)], 'size と同じ') },
  pcaDim: { kind: 'select', choices: choices(['4', '8', '16', '32', '64', '128', '256']) },
  positionWeight: {
    kind: 'slider',
    choices: choices(['0', '0.1', '0.2', '0.5', '1', '2', '5', '10', '20', '50', '100']),
  },
  blur: {
    kind: 'slider',
    choices: choices([
      '0.01',
      '0.02',
      '0.03',
      '0.05',
      '0.07',
      '0.1',
      '0.15',
      '0.2',
      '0.3',
      '0.5',
      '0.7',
      '1',
      '2',
      '5',
      '10',
    ]),
  },
  scaling: {
    kind: 'slider',
    choices: choices(['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '0.95']),
  },
  reachX: { kind: 'slider', choices: choices(REACH, 'balanced') },
  reachY: { kind: 'slider', choices: choices(REACH, 'balanced') },
  threshold: {
    kind: 'select',
    choices: choices(['0.01', '0.001', '0.0001', '0.00001', '0.000001', ''], '打ち切りなし'),
  },
  innerIterations: { kind: 'select', choices: choices(['1', '5', '10', '20', '50', '100']) },
  topK: { kind: 'slider', choices: choices(ints(1, 10)) },
  minWeight: { kind: 'select', choices: choices(['0', '0.00001', '0.0001', '0.001', '0.01', '0.1']) },
}

export type ParamErrors = Partial<Record<NumericKey, string>>

/** 「2 以上 64 以下の整数」のような、許される範囲の説明。 */
export function describeRange(spec: NumberSpec): string {
  const noun = spec.integer ? '整数' : '数'
  const { min, max } = spec
  const lowerOpen = spec.minInclusive === false
  const upperOpen = spec.maxInclusive === false
  if (min !== undefined && max !== undefined) {
    return `${min} ${lowerOpen ? 'より大きく' : '以上'} ${max} ${upperOpen ? '未満' : '以下'}の${noun}`
  }
  if (min !== undefined) return lowerOpen ? `${min} より大きい${noun}` : `${min} 以上の${noun}`
  if (max !== undefined) return upperOpen ? `${max} 未満の${noun}` : `${max} 以下の${noun}`
  return noun
}

const NUMBER_RE = /^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$/

export type ParsedNumber = { ok: true; value: number | null } | { ok: false; error: string }

/** 入力文字列を数値にする。空欄は `optional` のときだけ null、それ以外は範囲の説明つきのエラー。 */
export function parseNumber(text: string, spec: NumberSpec): ParsedNumber {
  const s = text.trim()
  const range = describeRange(spec)
  if (s === '') return spec.optional ? { ok: true, value: null } : { ok: false, error: `${range}を入力してください` }
  if (!NUMBER_RE.test(s)) return { ok: false, error: `${range}を入力してください` }
  const value = Number(s)
  if (!Number.isFinite(value) || (spec.integer && !Number.isInteger(value))) {
    return { ok: false, error: `${range}を入力してください` }
  }
  const { min, max } = spec
  const belowMin = min !== undefined && (spec.minInclusive === false ? value <= min : value < min)
  const aboveMax = max !== undefined && (spec.maxInclusive === false ? value >= max : value > max)
  if (belowMin || aboveMax) return { ok: false, error: `${range}にしてください` }
  return { ok: true, value }
}

/** 画像 ID を除いたリクエスト本体 (`POST /api/match` の image_a / image_b 以外)。 */
export type MatchParams = Required<Pick<MatchRequest, 'patch' | 'feature' | 'ot' | 'output'>>

export type ParsedParams = { ok: true; params: MatchParams } | { ok: false; errors: ParamErrors }

/**
 * ドラフトを検証してリクエスト本体にする。すべての項目を明示して送る (省略時の既定値にサーバ側の変更が
 * 紛れ込まないように。結果 JSON にも実際に使った値が残る)。
 */
export function parseParams(draft: ParamDraft): ParsedParams {
  const errors: ParamErrors = {}
  const values = {} as Record<NumericKey, number | null>
  for (const key of Object.keys(NUMBER_SPECS) as NumericKey[]) {
    const parsed = parseNumber(draft[key], NUMBER_SPECS[key])
    if (parsed.ok) values[key] = parsed.value
    else errors[key] = parsed.error
  }
  if (Object.keys(errors).length > 0) return { ok: false, errors }

  // 上のループで、必須の項目は number であることを確認済み
  const num = (key: NumericKey) => values[key] as number
  return {
    ok: true,
    params: {
      patch: { size: num('patchSize'), stride: values.stride },
      feature: {
        type: draft.featureType,
        pca_dim: num('pcaDim'),
        normalize: draft.normalize,
        position_weight: num('positionWeight'),
      },
      ot: {
        blur: num('blur'),
        scaling: num('scaling'),
        half_cost: draft.halfCost,
        reach_x: values.reachX,
        reach_y: values.reachY,
        threshold: values.threshold,
        inner_iterations: num('innerIterations'),
        backend: draft.backend === '' ? null : draft.backend,
      },
      output: {
        top_k: num('topK'),
        min_weight: num('minWeight'),
        compute_divergence: draft.computeDivergence,
      },
    },
  }
}

/** 2 つのリクエスト本体が同じ内容か。結果が現在の設定で得られたものかの判定に使う。 */
export function sameParams(a: MatchParams | null, b: MatchParams | null): boolean {
  if (a === null || b === null) return a === b
  return JSON.stringify(a) === JSON.stringify(b)
}

/** パッチ設定だけを取り出す (格子のプレビュー用。他の項目が不正でも、ここが正しければ求まる)。 */
export function parsePatch(draft: ParamDraft): { size: number; stride: number } | null {
  const size = parseNumber(draft.patchSize, NUMBER_SPECS.patchSize)
  const stride = parseNumber(draft.stride, NUMBER_SPECS.stride)
  if (!size.ok || size.value === null || !stride.ok) return null
  return { size: size.value, stride: stride.value ?? size.value }
}

/** リクエストからパラメータ部分を取り出す。`parseParams` が作ったものなら常に全項目がある。 */
export function paramsOf(request: MatchRequest): MatchParams | null {
  const { patch, feature, ot, output } = request
  return patch && feature && ot && output ? { patch, feature, ot, output } : null
}
