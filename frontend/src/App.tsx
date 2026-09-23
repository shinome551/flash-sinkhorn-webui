import { useQueryClient } from '@tanstack/react-query'
import { useEffect, useMemo, useRef, useState } from 'react'
import { ApiError, imageExists } from './api/client.ts'
import type { ImageUploadResponse, MatchRequest, SamplePairResponse } from './api/types.ts'
import { HealthBadge } from './components/HealthBadge.tsx'
import { MatchStage } from './components/MatchStage.tsx'
import { ModeSelector } from './components/ModeSelector.tsx'
import { ParamPanel } from './components/ParamPanel.tsx'
import { SamplePicker } from './components/SamplePicker.tsx'
import { ServerNotice } from './components/ServerNotice.tsx'
import { StatsPanel } from './components/StatsPanel.tsx'
import { useHealth } from './hooks/useHealth.ts'
import { useMatch } from './hooks/useMatch.ts'
import { describeError } from './lib/errors.ts'
import { computeGrid } from './lib/grid.ts'
import type { Assignment } from './lib/assignment.ts'
import type { VizMode } from './lib/modes.ts'
import type { WeightNorm } from './lib/overlay.ts'
import { featureUnavailable, flashUnavailable, runBlocker as findRunBlocker } from './lib/preflight.ts'
import { DEFAULT_DRAFT, paramsOf, parseParams, parsePatch, sameParams, type ParamDraft } from './lib/params.ts'

/** flow の矢印を間引く既定の間隔 (2 パッチおき)。 */
const DEFAULT_FLOW_STEP = 2

function App() {
  const [imageA, setImageA] = useState<ImageUploadResponse | null>(null)
  const [imageB, setImageB] = useState<ImageUploadResponse | null>(null)
  const [expiredA, setExpiredA] = useState(false)
  const [expiredB, setExpiredB] = useState(false)
  const [showGrid, setShowGrid] = useState(true)
  const [draft, setDraft] = useState<ParamDraft>(DEFAULT_DRAFT)
  const [mode, setMode] = useState<VizMode>('hover')
  const [assignment, setAssignment] = useState<Assignment>('soft')
  const [weightNorm, setWeightNorm] = useState<WeightNorm>('relative')
  const [flowStep, setFlowStep] = useState(DEFAULT_FLOW_STEP)
  const [flowFade, setFlowFade] = useState(true)
  const healthQuery = useHealth()
  // 再取得に失敗している間は、直前の値 (data) を信用しない
  const health = healthQuery.error ? undefined : healthQuery.data
  const maxBytes = health?.limits.max_upload_bytes
  const queryClient = useQueryClient()

  const match = useMatch({
    onError: (error, req) => {
      if (!(error instanceof ApiError)) return
      if (error.code === 'IMAGE_NOT_FOUND') void dropMissingImages(req)
      // 停止を検知したら、15 秒の定期確認を待たずにバッジと案内を切り替える
      if (error.code === 'NETWORK_ERROR') void queryClient.invalidateQueries({ queryKey: ['health'] })
    },
  })

  // 確認の await 中に画像が差し替えられることがあるので、最新の値を ref で参照する
  const imagesRef = useRef({ a: imageA, b: imageB })
  useEffect(() => {
    imagesRef.current = { a: imageA, b: imageB }
  })

  /** どちらの画像が TTL 切れか分からないので、両方を取得してみて、無い方だけを外す。 */
  async function dropMissingImages(req: MatchRequest) {
    const [okA, okB] = await Promise.all([imageExists(req.image_a), imageExists(req.image_b)])
    // 確認中に差し替えられた画像には期限切れの案内を出さない
    if (!okA && imagesRef.current.a?.image_id === req.image_a) {
      setImageA(null)
      setExpiredA(true)
    }
    if (!okB && imagesRef.current.b?.image_id === req.image_b) {
      setImageB(null)
      setExpiredB(true)
    }
  }

  // 入力画像が変わったら、実行中のリクエストと古い結果は捨てる。
  const changeImageA = (image: ImageUploadResponse | null) => {
    setImageA(image)
    setExpiredA(false)
    match.reset()
  }
  const changeImageB = (image: ImageUploadResponse | null) => {
    setImageB(image)
    setExpiredB(false)
    match.reset()
  }
  const loadSample = (pair: SamplePairResponse) => {
    changeImageA(pair.a)
    changeImageB(pair.b)
  }

  const parsed = useMemo(() => parseParams(draft), [draft])
  const patch = useMemo(() => parsePatch(draft), [draft])
  const previewGridA = useMemo(
    () => (imageA && patch ? computeGrid(imageA.width, imageA.height, patch.size, patch.stride) : null),
    [imageA, patch],
  )
  const previewGridB = useMemo(
    () => (imageB && patch ? computeGrid(imageB.width, imageB.height, patch.size, patch.stride) : null),
    [imageB, patch],
  )

  const runBlocker = findRunBlocker({
    hasImageA: imageA !== null,
    hasImageB: imageB !== null,
    paramsValid: parsed.ok,
    gridA: previewGridA,
    gridB: previewGridB,
    health,
    backend: draft.backend,
    featureType: draft.featureType,
  })
  const canRun = runBlocker === undefined
  const run = () => {
    if (imageA && imageB && parsed.ok) {
      match.run({ image_a: imageA.image_id, image_b: imageB.image_id, ...parsed.params })
    }
  }

  // 表示中の結果が、現在の入力欄と別の設定で得たものか (入力欄が不正な間も「別」とみなす)。
  const stale = match.request !== null && !sameParams(parsed.ok ? parsed.params : null, paramsOf(match.request))

  // 期限切れの画像は外した上でスロット側に案内を出すので、同じ内容のバナーは重ねない。
  const isNotFound = match.error instanceof ApiError && match.error.code === 'IMAGE_NOT_FOUND'
  const error = match.error && !(isNotFound && (expiredA || expiredB)) ? describeError(match.error) : null

  // ヘッダーとパラメータパネルの両方に置く (画像を見ながらでも、パラメータを変えた直後でも押せるように)
  const runButton = (
    <RunButton
      isPending={match.isPending}
      onRun={run}
      onCancel={match.cancel}
      disabled={!canRun}
      blocker={runBlocker}
    />
  )

  return (
    <div className="min-h-screen bg-slate-900 text-slate-100">
      <header className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-800 px-4 py-3 sm:px-6">
        <h1 className="text-lg font-bold">flash-sinkhorn-webui</h1>
        <div className="flex flex-wrap items-center gap-4">
          <label className="flex cursor-pointer items-center gap-2 text-xs text-slate-300">
            <input type="checkbox" checked={showGrid} onChange={(e) => setShowGrid(e.target.checked)} />
            パッチ格子{patch && ` (${patch.size}px)`}
          </label>
          <HealthBadge />
          {runButton}
        </div>
      </header>

      <main className="mx-auto max-w-7xl space-y-4 p-4 sm:p-6">
        <ServerNotice />
        {error && (
          <div className="rounded border border-red-800 bg-red-950/60 px-3 py-2 text-sm text-red-300" role="alert">
            <p className="font-medium">{error.title}</p>
            {(error.detail || error.hint) && (
              <p className="mt-1 text-xs text-red-300/80">{[error.detail, error.hint].filter(Boolean).join(' ')}</p>
            )}
          </div>
        )}
        <div className="flex flex-wrap items-center justify-between gap-x-6 gap-y-2">
          <SamplePicker onLoad={loadSample} />
          {runBlocker && !match.isPending && (
            <p
              className={`text-xs ${imageA && imageB ? 'text-amber-300' : 'text-slate-400'}`}
              role="status"
              data-testid="run-blocker"
            >
              {runBlocker}
            </p>
          )}
        </div>
        <ModeSelector
          assignment={assignment}
          onAssignmentChange={setAssignment}
          mode={mode}
          onModeChange={setMode}
          weightNorm={weightNorm}
          onWeightNormChange={setWeightNorm}
          flowStep={flowStep}
          onFlowStepChange={setFlowStep}
          flowFade={flowFade}
          onFlowFadeChange={setFlowFade}
        />
        <MatchStage
          imageA={imageA}
          imageB={imageB}
          onChangeA={changeImageA}
          onChangeB={changeImageB}
          expiredA={expiredA}
          expiredB={expiredB}
          maxBytes={maxBytes}
          previewGridA={previewGridA}
          previewGridB={previewGridB}
          showGrid={showGrid}
          result={match.result}
          request={match.request}
          isPending={match.isPending}
          assignment={assignment}
          mode={mode}
          weightNorm={weightNorm}
          flowStep={flowStep}
          flowFade={flowFade}
        />
        <div className="grid grid-cols-1 items-start gap-4 lg:grid-cols-2">
          <ParamPanel
            draft={draft}
            errors={parsed.ok ? {} : parsed.errors}
            onChange={(change) => setDraft((d) => ({ ...d, ...change }))}
            onReset={() => setDraft(DEFAULT_DRAFT)}
            runButton={runButton}
            runBlocker={match.isPending ? undefined : runBlocker}
            gridA={previewGridA}
            gridB={previewGridB}
            maxPatches={health?.limits.max_patches}
            featureTypes={health?.feature_types}
            featureError={
              health && featureUnavailable(health, draft.featureType)
                ? 'サーバに extra deep (timm) が入っていません (uv sync --extra deep)'
                : undefined
            }
            backendError={
              health && flashUnavailable(health, draft.backend)
                ? 'CUDA か flash-sinkhorn が無いので flash は使えません (auto / dense を選んでください)'
                : undefined
            }
          />
          <StatsPanel result={match.result} previous={match.previousStats} stale={stale} isPending={match.isPending} />
        </div>
      </main>
    </div>
  )
}

function RunButton({
  isPending,
  onRun,
  onCancel,
  disabled,
  blocker,
}: {
  isPending: boolean
  onRun: () => void
  onCancel: () => void
  disabled: boolean
  blocker?: string
}) {
  return isPending ? (
    <button
      type="button"
      onClick={onCancel}
      className="rounded bg-slate-700 px-4 py-1.5 text-sm font-medium text-slate-100 hover:bg-slate-600"
    >
      中断
    </button>
  ) : (
    <button
      type="button"
      onClick={onRun}
      disabled={disabled}
      title={blocker}
      className="rounded bg-sky-600 px-4 py-1.5 text-sm font-medium text-white hover:bg-sky-500 disabled:cursor-not-allowed disabled:bg-slate-700 disabled:text-slate-400"
    >
      実行
    </button>
  )
}

export default App
