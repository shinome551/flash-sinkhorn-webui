// ソフト (エントロピー正則化した計画) / ハード (c-transform の argmin) の表示切替 (SPEC 3.6)。
// ハードは結果を「対応先 1 件・重み 1」の形に読み替えるだけなので、再実行は要らず、描画側 (overlay / flow /
// heatmap / PNG 書き出し) はどちらでも同じコードを通る。

import type { MatchResponse } from '../api/types.ts'

export type Assignment = 'soft' | 'hard'

/**
 * 表示に使う結果。hard では targets を hard.j の 1 件 (weight 1)、displacement を hard.displacement、
 * col_mass を hard_col_mass に差し替え、row_mass を 1 (各パッチがちょうど 1 回送る) にする。
 * confidence / entropy はソフト計画の量のまま (どちらでも意味がある)。
 */
export function viewResult(result: MatchResponse, assignment: Assignment): MatchResponse {
  if (assignment === 'soft') return result
  return {
    ...result,
    matches: result.matches.map((m) => ({
      ...m,
      row_mass: 1,
      displacement: m.hard.displacement,
      targets: [{ j: m.hard.j, weight: 1, cost: m.hard.cost }],
    })),
    col_mass: result.hard_col_mass,
  }
}
