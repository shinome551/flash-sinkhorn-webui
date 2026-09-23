import { useId, type ReactNode } from 'react'
import type { BackendChoice, FeatureType, GridInfo, NormalizeMode } from '../api/types.ts'
import { describeRange, NUMBER_SPECS, type NumericKey, type ParamDraft, type ParamErrors } from '../lib/params.ts'
import { patchCount } from '../lib/grid.ts'
import { HelpTip } from './HelpTip.tsx'

interface Props {
  draft: ParamDraft
  errors: ParamErrors
  onChange: (patch: Partial<ParamDraft>) => void
  onReset: () => void
  /** 現在のパッチ設定での格子 (パッチ数の目安)。画像が未選択なら null、設定が不正でも null */
  gridA: GridInfo | null
  gridB: GridInfo | null
  maxPatches?: number
  /** サーバで使える特徴の種類 (health)。未取得なら undefined で、全種類を選べる */
  featureTypes?: FeatureType[]
  /** 選んだ特徴の種類がサーバで使えないときの説明 */
  featureError?: string
  /** バックエンドの選択がサーバで使えないとき (CUDA が無いのに flash など) の説明 */
  backendError?: string
}

const HELP = {
  patchSize:
    'パッチ (正方形のタイル) の一辺 [px]。小さいほど細かい対応が得られますが、パッチ数 N は (画像面積 / size²) で増え、計算時間と N×M の輸送計画が大きくなります。',
  stride:
    'パッチを取る間隔 [px]。空欄なら size と同じ (重ならない格子)。size より小さいとパッチが重なって N が増え、大きいと隙間ができます。',
  featureType:
    'パッチをどんなベクトルにして距離を測るか。raw = 画素をそのまま (3·size² 次元)、pca = raw を A・B 共通の基底で圧縮 (既定)、color = 平均 RGB のみ (3 次元、軽量なデモ用)、dinov2 = 自己教師あり ViT (DINOv2 ViT-S/14) の中間表現 (384 次元)。dinov2 は見た目より「何が写っているか」で対応するので、左右反転や別の写真どうしに強い一方、単純な平行移動の位置精度は pca より落ちます。画像は stride 1 つがトークン 1 個 (14px) になるよう拡大縮小してモデルに通すため、パッチサイズを変えるとモデルが見る解像度も変わります。バックエンドに extra deep が必要で、初回は重みの読み込みに数秒〜十数秒かかります。',
  pcaDim:
    'pca のときの次元数。raw の次元数か N+M を超える分は切り詰められ、警告が出ます。他の特徴 (dinov2 を含む) では使われません。',
  normalize:
    '特徴のスケール調整。zscore = 次元ごとに標準化して 1/√d 倍 (既定。blur を次元数に依らず選べる)、l2 = ベクトルを単位長に、none = そのまま (画素スケールなので blur を手動で調整する必要があります)。',
  positionWeight:
    '0 より大きくすると、正規化したパッチ中心 (x/W, y/H) を λ 倍して特徴に連結します。近い位置どうしが対応しやすくなる空間的な事前分布で、大きいほど位置が優先されます。',
  blur: 'エントロピー正則化の強さで、ε = blur² (コストの尺度に対する滑らかさ)。小さいほど 1 対 1 に近いシャープな対応になりますが、収束が遅くなり行質量の誤差の警告が出やすくなります。大きいほど対応が拡散して確信度が下がります。zscore の特徴なら 0.05〜0.3 が目安です。',
  scaling:
    'ε スケジューリングの縮小率 (0〜1)。ε を大きい値から blur² まで、反復ごとにこの倍率で下げていきます。1 に近いほどゆっくり下げるので安定しますが、反復が増えます。',
  halfCost:
    'コストを ||x−y||²/2 にします (既定は ||x−y||²)。同じ blur でも実質の正則化が変わるので、他の実装の規約に合わせたいときに使います。',
  reachX:
    'A 側の質量保存を緩めます (unbalanced OT)。空欄 = balanced で、A の各パッチは質量 1/N をすべて B へ送ります。値を入れると、コストが reach² 程度を超える (特徴が遠すぎる) 輸送は質量ごと諦めてよくなり、対応先の無いパッチが許容されます。小さいほど緩く、zscore の特徴では 0.2〜1 が目安です (0.1 以下ではほとんど何も運ばれません)。reach_x だけの設定 (semi-unbalanced) では B 側の質量を満たすため、対応しやすい A パッチが 1/N より多く送ることもあります。送出量はカバレッジモードの A 側と hover の表示で確認できます。',
  reachY:
    'B 側の質量保存を緩めます。空欄 = balanced。値を入れると、B のパッチが受け取る質量が 1/M から偏ってよくなり、カバレッジモードの B 側で受け取り量の偏りとして見えます。値の目安と注意は reach_x と同じです。unbalanced では運ばれる総質量が 1 未満になるので、ダイバージェンスは balanced の値と比べられません。',
  threshold:
    '収束判定 (ε 相対)。inner_iterations 回ごとのポテンシャルの最大変化量が、この値 × ε 未満になったら打ち切ります。空欄で打ち切りなし (サーバの反復上限まで実行)。収束しなかった場合は警告が出ます。',
  innerIterations: '収束を確認する間隔 (反復回数)。大きいほど確認のオーバーヘッドは減りますが、打ち切りが遅れます。',
  backend:
    '空欄 = サーバ設定。auto = CUDA があれば flash-sinkhorn、無ければ dense。flash = flash-sinkhorn を明示 (CUDA が無いとエラー)。dense = N×M のコスト行列を実体化する参照実装 (CPU / GPU)。',
  topK: '各パッチについて返す対応先の数。hover の接続線や B 側の枠はこの数だけ描かれます。',
  minWeight: '条件付き確率がこの値未満の対応先は結果から外します (0 件になることもあります)。',
  computeDivergence:
    'Sinkhorn ダイバージェンス (debias 版。自己項を引いた 2 画像間の距離) を別途計算して表示します。輸送計画そのものは debias していないポテンシャルから作るので、切っても対応結果は変わりません。reach を設定した (unbalanced) 実行の値は balanced の値と比べられません。',
} as const

const FEATURE_TYPES: FeatureType[] = ['pca', 'raw', 'color', 'dinov2']
const NORMALIZE_MODES: NormalizeMode[] = ['zscore', 'l2', 'none']
const BACKENDS: { value: '' | BackendChoice; label: string }[] = [
  { value: '', label: 'サーバ設定' },
  { value: 'auto', label: 'auto' },
  { value: 'flash', label: 'flash' },
  { value: 'dense', label: 'dense' },
]

const inputClass =
  'w-28 rounded border bg-slate-800 px-2 py-1 text-right text-sm tabular-nums text-slate-100 focus-visible:outline-2 focus-visible:outline-sky-400 disabled:cursor-not-allowed disabled:opacity-40'

export function ParamPanel({
  draft,
  errors,
  onChange,
  onReset,
  gridA,
  gridB,
  maxPatches,
  featureTypes,
  featureError,
  backendError,
}: Props) {
  const num =
    (key: NumericKey, options: { disabled?: boolean; placeholder?: string } = {}) =>
    (id: string) => (
      <NumberInput
        id={id}
        value={draft[key]}
        onChange={(v) => onChange({ [key]: v })}
        error={errors[key]}
        placeholder={options.placeholder}
        disabled={options.disabled}
        range={describeRange(NUMBER_SPECS[key])}
      />
    )

  return (
    <section aria-label="パラメータ" className="min-w-0 rounded border border-slate-800 bg-slate-900/60 p-4">
      <div className="mb-3 flex items-baseline justify-between gap-2">
        <h2 className="text-sm font-semibold text-slate-200">パラメータ</h2>
        <button type="button" onClick={onReset} className="text-xs text-slate-400 underline hover:text-slate-200">
          既定値に戻す
        </button>
      </div>
      <p className="mb-3 text-xs text-slate-500">変更は「実行」を押すまで反映されません。</p>

      <div className="space-y-4">
        <Group title="パッチ">
          <Field label="size [px]" help={HELP.patchSize} error={errors.patchSize}>
            {num('patchSize')}
          </Field>
          <Field label="stride [px]" help={HELP.stride} error={errors.stride}>
            {num('stride', { placeholder: 'size と同じ' })}
          </Field>
          <PatchCounts gridA={gridA} gridB={gridB} maxPatches={maxPatches} />
        </Group>

        <Group title="特徴">
          <Field label="type" help={HELP.featureType} error={featureError}>
            {(id) => (
              <Select
                id={id}
                value={draft.featureType}
                onChange={(v) => onChange({ featureType: v })}
                options={FEATURE_TYPES.map((v) => {
                  const missing = featureTypes !== undefined && !featureTypes.includes(v)
                  return {
                    value: v,
                    label: missing ? `${v} (未導入)` : v,
                    disabled: missing && v !== draft.featureType,
                  }
                })}
              />
            )}
          </Field>
          <Field label="pca_dim" help={HELP.pcaDim} error={errors.pcaDim}>
            {num('pcaDim', { disabled: draft.featureType !== 'pca' })}
          </Field>
          <Field label="normalize" help={HELP.normalize}>
            {(id) => (
              <Select
                id={id}
                value={draft.normalize}
                onChange={(v) => onChange({ normalize: v })}
                options={NORMALIZE_MODES.map((v) => ({ value: v, label: v }))}
              />
            )}
          </Field>
          <Field label="position_weight" help={HELP.positionWeight} error={errors.positionWeight}>
            {num('positionWeight')}
          </Field>
        </Group>

        <Group title="最適輸送 (OT)">
          <Field label="blur" help={HELP.blur} error={errors.blur}>
            {num('blur')}
          </Field>
          <Field label="scaling" help={HELP.scaling} error={errors.scaling}>
            {num('scaling')}
          </Field>
          <Field label="half_cost" help={HELP.halfCost}>
            {(id) => <Check id={id} checked={draft.halfCost} onChange={(v) => onChange({ halfCost: v })} />}
          </Field>
          <Field label="reach_x" help={HELP.reachX} error={errors.reachX}>
            {num('reachX', { placeholder: 'balanced' })}
          </Field>
          <Field label="reach_y" help={HELP.reachY} error={errors.reachY}>
            {num('reachY', { placeholder: 'balanced' })}
          </Field>
          <Field label="threshold" help={HELP.threshold} error={errors.threshold}>
            {num('threshold', { placeholder: '打ち切りなし' })}
          </Field>
          <Field label="inner_iterations" help={HELP.innerIterations} error={errors.innerIterations}>
            {num('innerIterations')}
          </Field>
          <Field label="backend" help={HELP.backend} error={backendError}>
            {(id) => (
              <Select id={id} value={draft.backend} onChange={(v) => onChange({ backend: v })} options={BACKENDS} />
            )}
          </Field>
        </Group>

        <Group title="出力">
          <Field label="top_k" help={HELP.topK} error={errors.topK}>
            {num('topK')}
          </Field>
          <Field label="min_weight" help={HELP.minWeight} error={errors.minWeight}>
            {num('minWeight')}
          </Field>
          <Field label="compute_divergence" help={HELP.computeDivergence}>
            {(id) => (
              <Check id={id} checked={draft.computeDivergence} onChange={(v) => onChange({ computeDivergence: v })} />
            )}
          </Field>
        </Group>
      </div>
    </section>
  )
}

function Group({ title, children }: { title: string; children: ReactNode }) {
  return (
    <fieldset className="min-w-0 space-y-2 border-t border-slate-800 pt-3">
      <legend className="pr-2 text-xs font-semibold uppercase tracking-wide text-slate-400">{title}</legend>
      {children}
    </fieldset>
  )
}

/** ラベル + ツールチップ + 入力欄 + エラー。入力欄は `useId` の id を受け取って `label` と結びつける。 */
function Field({
  label,
  help,
  error,
  children,
}: {
  label: string
  help: string
  error?: string
  children: (id: string) => ReactNode
}) {
  const id = useId()
  return (
    <div>
      <div className="flex items-center justify-between gap-3">
        <span className="flex min-w-0 items-center gap-1.5 text-sm text-slate-300">
          <label htmlFor={id} className="truncate font-mono text-xs">
            {label}
          </label>
          <HelpTip text={help} />
        </span>
        {children(id)}
      </div>
      {error && (
        <p className="mt-1 text-right text-xs text-red-400" role="alert">
          {error}
        </p>
      )}
    </div>
  )
}

function NumberInput({
  id,
  value,
  onChange,
  error,
  placeholder,
  disabled,
  range,
}: {
  id: string
  value: string
  onChange: (value: string) => void
  error?: string
  placeholder?: string
  disabled?: boolean
  range: string
}) {
  return (
    <input
      id={id}
      type="text"
      inputMode="decimal"
      value={value}
      placeholder={placeholder}
      disabled={disabled}
      title={range}
      aria-invalid={error ? true : undefined}
      onChange={(e) => onChange(e.target.value)}
      className={`${inputClass} ${error ? 'border-red-500' : 'border-slate-700'}`}
    />
  )
}

function Select<T extends string>({
  id,
  value,
  onChange,
  options,
}: {
  id: string
  value: T
  onChange: (value: T) => void
  options: { value: T; label: string; disabled?: boolean }[]
}) {
  return (
    <select
      id={id}
      value={value}
      onChange={(e) => onChange(e.target.value as T)}
      className="w-28 rounded border border-slate-700 bg-slate-800 px-2 py-1 text-sm text-slate-100 focus-visible:outline-2 focus-visible:outline-sky-400"
    >
      {options.map((o) => (
        <option key={o.value} value={o.value} disabled={o.disabled}>
          {o.label}
        </option>
      ))}
    </select>
  )
}

function Check({ id, checked, onChange }: { id: string; checked: boolean; onChange: (checked: boolean) => void }) {
  return (
    <input
      id={id}
      type="checkbox"
      checked={checked}
      onChange={(e) => onChange(e.target.checked)}
      className="h-4 w-4 accent-sky-500"
    />
  )
}

/** 現在のパッチ設定でのパッチ数。上限を超えるなら実行前に気づけるよう警告色にする。 */
function PatchCounts({
  gridA,
  gridB,
  maxPatches,
}: {
  gridA: GridInfo | null
  gridB: GridInfo | null
  maxPatches?: number
}) {
  if (!gridA && !gridB) return null
  const item = (name: string, grid: GridInfo | null) => {
    if (!grid) return null
    const n = patchCount(grid)
    const over = maxPatches !== undefined && n > maxPatches
    return (
      <span className={over ? 'text-amber-400' : undefined}>
        {name}: {grid.rows}×{grid.cols} = {n}
        {over && ` (上限 ${maxPatches} 超過)`}
      </span>
    )
  }
  return (
    <p className="flex flex-wrap justify-end gap-x-3 text-xs text-slate-400">
      {item('A', gridA)}
      {item('B', gridB)}
    </p>
  )
}
