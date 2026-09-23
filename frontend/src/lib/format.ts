// 数値の表示用整形。

/** 有効数字 `digits` 桁 (既定 3)。極端に小さい・大きい値は指数表記。 */
export function formatNumber(value: number, digits = 3): string {
  if (!Number.isFinite(value)) return String(value)
  if (value === 0) return '0'
  const abs = Math.abs(value)
  if (abs < 1e-3 || abs >= 1e5) return value.toExponential(digits - 1)
  // toPrecision は 1e-3 台で指数表記に切り替わることがあるので、Number 経由で余分な 0 を落とす。
  return String(Number(value.toPrecision(digits)))
}

/** 符号つきの差分。0 は "±0"。 */
export function formatDelta(value: number, digits = 3): string {
  if (value === 0) return '±0'
  return `${value > 0 ? '+' : '−'}${formatNumber(Math.abs(value), digits)}`
}

/**
 * 範囲の両端を、幅が 2 桁以上読み取れる有効数字で整形する (3 桁から 6 桁)。
 * 狭い範囲 (例: 0.99981 〜 1.0002) が "1 〜 1" にならないようにする。
 */
export function formatRange(lo: number, hi: number): [string, string] {
  const magnitude = Math.max(Math.abs(lo), Math.abs(hi))
  const span = hi - lo
  let digits = 3
  if (span > 0 && magnitude > 0) {
    digits = Math.min(6, Math.max(3, Math.ceil(-Math.log10(span / magnitude)) + 1))
  }
  return [formatNumber(lo, digits), formatNumber(hi, digits)]
}
