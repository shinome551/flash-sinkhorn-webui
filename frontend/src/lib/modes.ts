// 可視化モード (SPEC 5.3)。モードごとの振る舞いの違いをここに集める。

export type VizMode = 'hover' | 'confidence' | 'entropy' | 'flow' | 'coverage' | 'pin'

export interface ModeInfo {
  id: VizMode
  label: string
  description: string
}

export const MODES: readonly ModeInfo[] = [
  {
    id: 'hover',
    label: 'hover',
    description: 'A のパッチにカーソルを置くと、B 側の対応先 (top-k) を重みに応じた濃さと接続線で表示します。',
  },
  {
    id: 'confidence',
    label: '確信度',
    description: 'A 側を確信度 (最大の条件付き確率) で着色します。低いほど対応先が拡散している (曖昧な) パッチです。',
  },
  {
    id: 'entropy',
    label: 'エントロピー',
    description: 'A 側を行エントロピーで着色します。高いほど対応先が分散しています。確信度と概ね逆の傾向です。',
  },
  {
    id: 'flow',
    label: 'フロー',
    description:
      'A 側の各パッチから、対応先の重心への変位 (ピクセル) を矢印で描きます。矢印は A の外へ出る部分を切り取ります。',
  },
  {
    id: 'coverage',
    label: 'カバレッジ',
    description:
      'A 側を送出量 (row_mass)、B 側を受け取り量 (col_mass) で着色します (どちらも 1.0 = 均等)。balanced ではほぼ一様で、reach を設定した (unbalanced) ときに偏りが見えます。送出量が 0 に近い A パッチは対応先を持たないパッチで、unbalanced の結果ではフローの矢印と pin の線もその分薄くなります。',
  },
  {
    id: 'pin',
    label: 'pin',
    description: 'クリック / Enter で A のパッチを固定し、複数パッチの対応先を色分けして同時に比較します。',
  },
]

/** クリックでのパッチ固定 (pin) を受け付けるモード。hover / flow では固定せず、固定済みの pin も描かない。 */
export function canPin(mode: VizMode): boolean {
  return mode !== 'hover' && mode !== 'flow'
}

/** hover 中のパッチの対応先 (B 側の枠と接続線) を描くモード。flow は矢印と重なるので A 側の枠だけ。 */
export function showsTargets(mode: VizMode): boolean {
  return mode !== 'flow'
}

export type HeatmapKind = 'confidence' | 'entropy' | 'coverage'

/** そのモードで塗るヒートマップ。無ければ null。 */
export function heatmapKind(mode: VizMode): HeatmapKind | null {
  return mode === 'confidence' || mode === 'entropy' || mode === 'coverage' ? mode : null
}
