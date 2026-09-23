// viridis 相当の連続カラーマップ。matplotlib の viridis を 9 点 (t = k/8) で間引き、区間を線形補間する。
// 知覚的に均一で、明度が t に対して単調に増える (0 = 暗い紫、1 = 明るい黄)。

export type RGB = readonly [number, number, number]

const VIRIDIS_STOPS: readonly RGB[] = [
  [0x44, 0x01, 0x54],
  [0x47, 0x2d, 0x7b],
  [0x3b, 0x52, 0x8b],
  [0x2c, 0x72, 0x8e],
  [0x21, 0x90, 0x8c],
  [0x27, 0xad, 0x81],
  [0x5d, 0xc8, 0x63],
  [0xaa, 0xdc, 0x32],
  [0xfd, 0xe7, 0x25],
]

/** t ∈ [0, 1] → RGB (各 0..255 の整数)。範囲外は端に丸め、NaN は 0 として扱う。 */
export function viridis(t: number): RGB {
  const x = Number.isNaN(t) ? 0 : Math.min(Math.max(t, 0), 1)
  const pos = x * (VIRIDIS_STOPS.length - 1)
  const k = Math.min(Math.floor(pos), VIRIDIS_STOPS.length - 2)
  const f = pos - k
  const lo = VIRIDIS_STOPS[k]
  const hi = VIRIDIS_STOPS[k + 1]
  return [
    Math.round(lo[0] + (hi[0] - lo[0]) * f),
    Math.round(lo[1] + (hi[1] - lo[1]) * f),
    Math.round(lo[2] + (hi[2] - lo[2]) * f),
  ]
}

/** CSS の `rgb(r, g, b)` 文字列。 */
export function viridisCss(t: number): string {
  const [r, g, b] = viridis(t)
  return `rgb(${r}, ${g}, ${b})`
}

/**
 * [lo, hi] を [0, 1] に正規化して viridis に通す。`hi <= lo` (定数場) は中間色 (0.5) を返す。
 * ヒートマップで値域を揃えるのに使う。
 */
export function viridisScaled(value: number, lo: number, hi: number): string {
  if (!(hi > lo)) return viridisCss(0.5)
  return viridisCss((value - lo) / (hi - lo))
}

/** CSS の左→右グラデーション (凡例の色バー用)。 */
export function viridisGradientCss(steps = 9): string {
  const stops = Array.from(
    { length: steps },
    (_, k) => `${viridisCss(k / (steps - 1))} ${Math.round((100 * k) / (steps - 1))}%`,
  )
  return `linear-gradient(to right, ${stops.join(', ')})`
}
