// A 側の送出量 (row_mass、a_i で割った値、1.0 = 質量をすべて送る)。unbalanced (reach_x) で意味を持つ。

import type { MatchResponse } from '../api/types.ts'

/** 送出量が 0 のときの不透明度の係数。0 にすると pin の線やフローの矢印が消えて、格子の抜けに見える。 */
export const MIN_MASS_OPACITY = 0.12

/**
 * unbalanced の実行か (API は balanced のときだけ `row_mass_error` を返す)。
 * balanced でも未収束だと row_mass は 1.0 から数十 % ずれるので、送出量で薄くするのは unbalanced のときだけ。
 */
export function isUnbalanced(result: MatchResponse): boolean {
  return result.stats.row_mass_error === null
}

/** 送出量 → 不透明度の係数 (MIN_MASS_OPACITY..1)。1.0 以上は 1、小さい値も見えるよう平方根を取る。 */
export function massOpacity(rowMass: number): number {
  const level = Math.sqrt(Math.min(Math.max(Number.isNaN(rowMass) ? 0 : rowMass, 0), 1))
  return MIN_MASS_OPACITY + (1 - MIN_MASS_OPACITY) * level
}
