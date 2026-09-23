// PNG 書き出しの凡例 (ラベル + 色バー + 値の範囲) の配置。幅が足りなければ 2 段に分け、それでも
// ラベルが入らなければ文字を縮める。描画は export.ts の drawLegend。

export interface LegendLayout {
  /** 1 = ラベルと色バーを横に並べる、2 = ラベルの下に色バー */
  rows: 1 | 2
  fontSize: number
  label: { x: number; y: number }
  /** 色バーの矩形。lo / hi の文字はその左右に置く (y は文字の中央) */
  bar: { x: number; y: number; width: number; height: number }
  lo: { x: number; y: number }
  hi: { x: number; y: number }
}

/**
 * `width` は出力画像の幅、`unit` は凡例 1 段の高さ [px]。`measure(text, fontSize)` は文字幅を返す
 * (canvas の measureText。テストでは近似)。y はすべて凡例の上端からの相対位置。
 */
export function layoutLegend(
  width: number,
  unit: number,
  texts: { label: string; lo: string; hi: string },
  measure: (text: string, fontSize: number) => number,
): LegendLayout {
  const pad = unit * 0.4
  const gap = unit * 0.15
  const barHeight = unit * 0.3
  const fullBar = unit * 6
  let fontSize = Math.round(unit * 0.4)
  const w = (text: string) => measure(text, fontSize)
  const rangeWidth = (bar: number) => w(texts.lo) + gap + bar + gap + w(texts.hi)

  // 1 段: [pad label  ...  lo ▭▭▭ hi pad]
  if (pad + w(texts.label) + unit * 0.6 + rangeWidth(fullBar) + pad <= width) {
    const barX = width - pad - w(texts.hi) - gap - fullBar
    const mid = unit / 2
    return {
      rows: 1,
      fontSize,
      label: { x: pad, y: mid },
      bar: { x: barX, y: mid - barHeight / 2, width: fullBar, height: barHeight },
      lo: { x: barX - gap, y: mid },
      hi: { x: barX + fullBar + gap, y: mid },
    }
  }

  // 2 段: 1 段目にラベル、2 段目に範囲。どちらも入らなければ文字を縮める (色バーは最低 unit 幅)
  const available = width - 2 * pad
  const needed = Math.max(w(texts.label), rangeWidth(unit))
  if (needed > available) fontSize = Math.max(1, Math.floor((fontSize * available) / needed))
  const bar = Math.max(0, Math.min(fullBar, available - rangeWidth(0)))
  const barX = pad + w(texts.lo) + gap
  const mid2 = unit * 1.5
  return {
    rows: 2,
    fontSize,
    label: { x: pad, y: unit / 2 },
    bar: { x: barX, y: mid2 - barHeight / 2, width: bar, height: barHeight },
    lo: { x: barX - gap, y: mid2 },
    hi: { x: barX + bar + gap, y: mid2 },
  }
}
