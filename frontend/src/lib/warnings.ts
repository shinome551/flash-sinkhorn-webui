// マッチ結果の警告 (MatchResponse.warnings) を日本語にする。コードは backend/app/services/notices.py。

import type { MatchWarning } from '../api/types.ts'

const percent = (v: unknown) => (typeof v === 'number' ? `${(v * 100).toFixed(1)}%` : '?')

/** 未知のコードや、params が欠けているときはサーバの英語メッセージをそのまま返す。 */
export function describeWarning(w: MatchWarning): string {
  const p = w.params
  switch (w.code) {
    case 'PCA_DIM_REDUCED':
      if (typeof p.requested !== 'number' || typeof p.actual !== 'number') break
      return `pca_dim を ${p.requested} から ${p.actual} に減らしました (パッチ数または raw の次元数が上限)。`
    case 'ROW_MASS_ERROR':
      if (typeof p.error !== 'number') break
      return (
        `行質量が一様な重みから最大 ${percent(p.error)} ずれています` +
        `${p.converged === false ? ' (反復上限で停止)' : ''}。` +
        '対応の重みは行ごとに正規化して表示するので可視化は成り立ちますが、周辺分布は近似です。blur を大きくすると改善します。'
      )
    case 'NOT_CONVERGED':
      if (typeof p.iterations !== 'number') break
      return `収束判定に達する前に反復上限 (${p.iterations} 回) で停止しました。結果は近似です。`
  }
  return w.message
}
