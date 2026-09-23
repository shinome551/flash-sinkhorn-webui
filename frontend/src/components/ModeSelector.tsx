import type { Assignment } from '../lib/assignment.ts'
import { MODES, showsTargets, type VizMode } from '../lib/modes.ts'
import type { WeightNorm } from '../lib/overlay.ts'

interface Props {
  assignment: Assignment
  onAssignmentChange: (assignment: Assignment) => void
  mode: VizMode
  onModeChange: (mode: VizMode) => void
  weightNorm: WeightNorm
  onWeightNormChange: (norm: WeightNorm) => void
  /** flow の矢印を間引く間隔 (パッチ数) */
  flowStep: number
  onFlowStepChange: (step: number) => void
  /** flow の矢印を確信度で薄くする */
  flowFade: boolean
  onFlowFadeChange: (fade: boolean) => void
}

export const FLOW_STEP_MAX = 8

/** 可視化モードの切替と、モードごとの表示オプション (重みの基準 / 矢印の間引き)。 */
const ASSIGNMENTS: { id: Assignment; label: string; description: string }[] = [
  {
    id: 'soft',
    label: 'ソフト',
    description:
      'エントロピー正則化した輸送計画の上位 k 件 (条件付き確率)。フローは分布の重心、カバレッジは row_mass / col_mass。',
  },
  {
    id: 'hard',
    label: 'ハード',
    description:
      'c-transform の argmin (j* = argmin_j [C_ij − g_j]) による 1 件の対応。一様重みではソフトの top-1 と同じ対応先で、フローはその中心を指し、カバレッジは選ばれた回数になります。再実行は不要です。',
  },
]

/** ハード割当で意味が変わるモードの補足 (説明文はソフト前提で書いてある)。 */
const HARD_NOTE: Partial<Record<VizMode, string>> = {
  flow: 'ハード割当では、重心ではなく argmin の対応先の中心を指します。',
  coverage:
    'ハード割当では、B 側だけを各パッチが選ばれた回数 × M/N (0 = どの A パッチからも選ばれない) で着色します (A の各パッチはちょうど 1 回ずつ送るため)。',
}

export function ModeSelector({
  assignment,
  onAssignmentChange,
  mode,
  onModeChange,
  weightNorm,
  onWeightNormChange,
  flowStep,
  onFlowStepChange,
  flowFade,
  onFlowFadeChange,
}: Props) {
  const current = MODES.find((m) => m.id === mode)
  return (
    <div className="space-y-2">
      <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
        <div
          role="radiogroup"
          aria-label="可視化モード"
          className="inline-flex flex-wrap gap-1 rounded-lg bg-slate-800 p-1"
        >
          {MODES.map((m) => (
            <label key={m.id} className="cursor-pointer">
              <input
                type="radio"
                name="viz-mode"
                value={m.id}
                checked={mode === m.id}
                onChange={() => onModeChange(m.id)}
                className="peer sr-only"
              />
              <span className="block rounded-md px-3 py-1 text-sm text-slate-300 hover:bg-slate-700 peer-checked:bg-sky-600 peer-checked:text-white peer-focus-visible:outline-2 peer-focus-visible:outline-sky-400">
                {m.label}
              </span>
            </label>
          ))}
        </div>

        <div role="radiogroup" aria-label="対応の種類" className="inline-flex gap-1 rounded-lg bg-slate-800 p-1">
          {ASSIGNMENTS.map((a) => (
            <label key={a.id} className="cursor-pointer" title={a.description}>
              <input
                type="radio"
                name="assignment"
                value={a.id}
                checked={assignment === a.id}
                onChange={() => onAssignmentChange(a.id)}
                className="peer sr-only"
              />
              <span className="block rounded-md px-3 py-1 text-sm text-slate-300 hover:bg-slate-700 peer-checked:bg-emerald-700 peer-checked:text-white peer-focus-visible:outline-2 peer-focus-visible:outline-sky-400">
                {a.label}
              </span>
            </label>
          ))}
        </div>

        {showsTargets(mode) && assignment === 'soft' && (
          <label className="flex items-center gap-2 text-xs text-slate-300">
            線の濃さの基準
            <select
              value={weightNorm}
              onChange={(e) => onWeightNormChange(e.target.value as WeightNorm)}
              className="rounded border border-slate-700 bg-slate-800 px-2 py-1 text-xs text-slate-100"
            >
              <option value="relative">相対 (top-1 = 最大)</option>
              <option value="absolute">絶対 (確率そのまま)</option>
            </select>
          </label>
        )}

        {mode === 'flow' && (
          <label className="flex items-center gap-2 text-xs text-slate-300">
            矢印の間引き
            <input
              type="range"
              min={1}
              max={FLOW_STEP_MAX}
              step={1}
              value={flowStep}
              onChange={(e) => onFlowStepChange(Number(e.target.value))}
              className="w-28 accent-sky-500"
            />
            <span className="w-24 tabular-nums text-slate-400">
              {flowStep === 1 ? 'すべて' : `${flowStep} パッチおき`}
            </span>
          </label>
        )}

        {mode === 'flow' && (
          <label
            className="flex cursor-pointer items-center gap-2 text-xs text-slate-300"
            title="確信度 (最大の条件付き確率) の低いパッチの矢印ほど薄く描きます。対応先の無い端のパッチは、拡散した分布の重心を指す長い矢印になりがちです。"
          >
            <input
              type="checkbox"
              checked={flowFade}
              onChange={(e) => onFlowFadeChange(e.target.checked)}
              className="accent-sky-500"
            />
            確信度で薄く
          </label>
        )}
      </div>
      {current && (
        <p className="text-xs leading-5 text-slate-400">
          {current.description}
          {assignment === 'hard' && HARD_NOTE[mode] && ` ${HARD_NOTE[mode]}`}
        </p>
      )}
    </div>
  )
}
