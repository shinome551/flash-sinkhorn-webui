import { useMemo, useRef, useState } from 'react'
import type { GridInfo, ImageUploadResponse, MatchRequest, MatchResponse } from '../api/types.ts'
import { useOverlayGeometry } from '../hooks/useOverlayGeometry.ts'
import { viewResult, type Assignment } from '../lib/assignment.ts'
import { buildFlow, FLOW_LABEL } from '../lib/flow.ts'
import { downloadBlob, renderPng, resultJson } from '../lib/export.ts'
import { fromIndex } from '../lib/grid.ts'
import { buildHeatmap } from '../lib/heatmap.ts'
import { canPin, heatmapKind, showsTargets, type VizMode } from '../lib/modes.ts'
import { buildOverlay, HOVER_COLOR, pinColor, type WeightNorm } from '../lib/overlay.ts'
import { togglePin, type Pin } from '../lib/selection.ts'
import { ColorBar } from './ColorBar.tsx'
import { ExportButtons } from './ExportButtons.tsx'
import { HeatmapLayer } from './HeatmapLayer.tsx'
import { ImageSlot } from './ImageSlot.tsx'
import { MatchOverlay } from './MatchOverlay.tsx'
import { PatchPicker } from './PatchPicker.tsx'

interface Props {
  imageA: ImageUploadResponse | null
  imageB: ImageUploadResponse | null
  onChangeA: (image: ImageUploadResponse | null) => void
  onChangeB: (image: ImageUploadResponse | null) => void
  expiredA: boolean
  expiredB: boolean
  maxBytes?: number
  /** マッチ結果が無い間に重ねるパッチ格子 (画像ごと)。結果があればその格子を使う */
  previewGridA: GridInfo | null
  previewGridB: GridInfo | null
  showGrid: boolean
  result: MatchResponse | null
  /** result を得たときのリクエスト (JSON 書き出し用) */
  request: MatchRequest | null
  isPending: boolean
  /** ソフト (計画の top-k) / ハード (c-transform の argmin) の表示。再実行せずに切り替わる */
  assignment: Assignment
  mode: VizMode
  weightNorm: WeightNorm
  /** flow の矢印を間引く間隔 (パッチ数) */
  flowStep: number
  /** flow の矢印を確信度で薄くする */
  flowFade: boolean
}

interface Selection {
  /** この選択がどの結果に対するものか。結果が変わったら選択は捨てる */
  forResult: MatchResponse | null
  active: number | null
  pins: Pin[]
}

/**
 * 画像 A・B を並べ、マッチ結果のヒートマップ / 矢印 / hover / pin と 2 画像をまたぐ接続線を重ねる。
 * 選択状態 (hover / pin) はここで持ち、モードを切り替えても保たれる。モードごとの違いは lib/modes.ts。
 */
export function MatchStage({
  imageA,
  imageB,
  onChangeA,
  onChangeB,
  expiredA,
  expiredB,
  maxBytes,
  previewGridA,
  previewGridB,
  showGrid,
  result,
  request,
  isPending,
  assignment,
  mode,
  weightNorm,
  flowStep,
  flowFade,
}: Props) {
  const stageRef = useRef<HTMLDivElement>(null)
  const boxARef = useRef<HTMLDivElement>(null)
  const boxBRef = useRef<HTMLDivElement>(null)

  const [selection, setSelection] = useState<Selection>({ forResult: result, active: null, pins: [] })
  if (selection.forResult !== result) {
    // 新しい結果が来たら (または消えたら) 選択を捨てる。render 中の state 調整は React 公式のパターン。
    setSelection({ forResult: result, active: null, pins: [] })
  }
  const { active, pins } = selection
  // 以降の描画はすべて view を使う。選択は元の result に紐づけ、ソフト / ハードを切り替えても保つ
  const view = useMemo(() => (result ? viewResult(result, assignment) : null), [result, assignment])

  const geometry = useOverlayGeometry(
    stageRef,
    boxARef,
    boxBRef,
    result?.grid_a.image_width,
    result?.grid_b.image_width,
    `${imageA?.image_id}/${imageB?.image_id}`,
  )

  const overlay = useMemo(
    () =>
      view && geometry.a && geometry.b
        ? buildOverlay({
            result: view,
            a: geometry.a,
            b: geometry.b,
            pins: canPin(mode) ? pins : [],
            active,
            weightNorm,
            showTargets: showsTargets(mode),
          })
        : null,
    [view, geometry, pins, active, mode, weightNorm],
  )

  const heatmap = useMemo(() => {
    const kind = heatmapKind(mode)
    return view && kind ? buildHeatmap(view, kind, assignment) : null
  }, [view, mode, assignment])

  // 矢印は多いので、hover のたびに作り直さない (geometry・密度・結果が変わったときだけ)。
  const flow = useMemo(
    () => (view && mode === 'flow' && geometry.a ? buildFlow(view, geometry.a, flowStep, flowFade) : null),
    [view, mode, geometry.a, flowStep, flowFade],
  )
  const legend = heatmap ?? (flow ? { ...flow, label: FLOW_LABEL } : null)

  const exportJson = () => {
    if (request && result) {
      downloadBlob(new Blob([resultJson(request, result)], { type: 'application/json' }), 'flash-sinkhorn-result.json')
    }
  }
  const exportPng = async () => {
    if (!imageA || !imageB || !view) return
    const blob = await renderPng({
      imageA,
      imageB,
      result: view,
      assignment,
      mode,
      pins,
      weightNorm,
      flowStep,
      flowFade,
      showGrid,
    })
    downloadBlob(blob, `flash-sinkhorn-${mode}${assignment === 'hard' ? '-hard' : ''}.png`)
  }

  const setActive = (i: number | null) => setSelection((s) => ({ ...s, active: i }))
  const onTogglePin = (i: number) => setSelection((s) => ({ ...s, pins: togglePin(s.pins, i) }))
  const clearPins = () => setSelection((s) => ({ ...s, pins: [] }))

  const gridA = result ? result.grid_a : previewGridA
  const gridB = result ? result.grid_b : previewGridB

  return (
    <>
      <div ref={stageRef} className="relative grid grid-cols-1 gap-6 lg:grid-cols-2">
        <ImageSlot
          label="画像 A"
          image={imageA}
          onChange={onChangeA}
          expired={expiredA}
          grid={showGrid ? gridA : null}
          maxBytes={maxBytes}
          boxRef={boxARef}
          overlay={(scale) =>
            result && (
              <>
                <HeatmapLayer heatmap={heatmap} side="a" displayScale={scale} />
                <PatchPicker
                  grid={result.grid_a}
                  displayScale={scale}
                  active={active}
                  onActiveChange={setActive}
                  onTogglePin={canPin(mode) ? onTogglePin : undefined}
                />
              </>
            )
          }
        />
        <ImageSlot
          label="画像 B"
          image={imageB}
          onChange={onChangeB}
          expired={expiredB}
          grid={showGrid ? gridB : null}
          maxBytes={maxBytes}
          boxRef={boxBRef}
          overlay={(scale) => <HeatmapLayer heatmap={heatmap} side="b" displayScale={scale} />}
        />
        {overlay && <MatchOverlay overlay={overlay} flow={flow} />}
        {isPending && (
          <div className="absolute inset-0 z-20 flex items-center justify-center rounded bg-slate-900/50" role="status">
            <span className="flex items-center gap-2 rounded-full bg-slate-800 px-4 py-2 text-sm text-slate-200 ring-1 ring-slate-600">
              <span
                className="h-3 w-3 animate-spin rounded-full border-2 border-slate-500 border-t-sky-400"
                aria-hidden
              />
              計算中…
            </span>
          </div>
        )}
      </div>

      {result && view && (
        <div className="mt-4 space-y-2">
          {legend && <ColorBar label={legend.label} lo={legend.lo} hi={legend.hi} />}
          <HoverInfo result={view} active={active} assignment={assignment} />
          {canPin(mode) && <PinBar result={result} pins={pins} onRemove={onTogglePin} onClear={clearPins} />}
          <ExportButtons disabled={!request} onJson={exportJson} onPng={exportPng} />
        </div>
      )}
    </>
  )
}

const HINT =
  'A の画像にカーソルを置く (矢印キーでも移動) と、B 側の対応先を表示します。クリック / Enter で固定 (pin・確信度・エントロピー・カバレッジのモード)、Esc で解除。'

function HoverInfo({
  result,
  active,
  assignment,
}: {
  result: MatchResponse
  active: number | null
  assignment: Assignment
}) {
  const match = active !== null ? result.matches[active] : undefined
  return (
    <p className="min-h-10 text-xs leading-5 text-slate-400">
      {match ? (
        <>
          <span className="inline-flex items-center gap-1 align-middle">
            <span className="inline-block h-2.5 w-2.5 rounded-sm" style={{ background: HOVER_COLOR }} aria-hidden />
            <span className="font-semibold text-slate-200">{describePatch(result.grid_a, match.i)}</span>
          </span>
          {' · '}確信度 {match.confidence.toFixed(2)} · エントロピー {match.entropy.toFixed(2)}
          {assignment === 'soft' && <> · 送出量 {match.row_mass.toFixed(2)}</>}
          {' · 変位 '}({signed(match.displacement[0])}, {signed(match.displacement[1])}) px
          {assignment === 'hard' ? (
            <>
              {' · ハード割当: '}
              {describePatch(result.grid_b, match.hard.j)} (コスト {match.hard.cost.toFixed(3)})
            </>
          ) : (
            <>
              {' · 対応先: '}
              {match.targets.length > 0
                ? match.targets.map((t) => `${describePatch(result.grid_b, t.j)} (${t.weight.toFixed(2)})`).join(', ')
                : 'なし (min_weight 未満)'}
            </>
          )}
        </>
      ) : (
        HINT
      )}
    </p>
  )
}

function PinBar({
  result,
  pins,
  onRemove,
  onClear,
}: {
  result: MatchResponse
  pins: Pin[]
  onRemove: (i: number) => void
  onClear: () => void
}) {
  if (pins.length === 0) return null
  return (
    <div className="flex flex-wrap items-center gap-2 text-xs">
      <span className="text-slate-400">固定:</span>
      {pins.map((p) => (
        <button
          key={p.i}
          type="button"
          onClick={() => onRemove(p.i)}
          title="クリックで固定を解除"
          className="inline-flex items-center gap-1.5 rounded-full bg-slate-800 px-2.5 py-1 text-slate-200 ring-1 ring-slate-700 hover:bg-slate-700"
        >
          <span className="h-2.5 w-2.5 rounded-sm" style={{ background: pinColor(p.slot) }} aria-hidden />
          {describePatch(result.grid_a, p.i)}
          <span aria-label="解除" className="text-slate-400">
            ×
          </span>
        </button>
      ))}
      <button type="button" onClick={onClear} className="text-slate-400 underline hover:text-slate-200">
        すべて解除
      </button>
    </div>
  )
}

const signed = (v: number) => `${v >= 0 ? '+' : '−'}${Math.abs(v).toFixed(1)}`

/** 「#33 (行 1, 列 1)」。行・列は 0 始まりではなく人間向けに 1 始まり。 */
function describePatch(grid: GridInfo, i: number): string {
  const { row, col } = fromIndex(grid, i)
  return `#${i} (行 ${row + 1}, 列 ${col + 1})`
}
