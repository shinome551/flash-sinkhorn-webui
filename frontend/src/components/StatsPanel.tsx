import type { ReactNode } from 'react'
import type { MatchResponse, MatchStats } from '../api/types.ts'
import { formatDelta, formatNumber } from '../lib/format.ts'
import { describeWarning } from '../lib/warnings.ts'

interface Props {
  result: MatchResponse | null
  /** 直前の実行の統計 (同じ画像ペアのとき)。あれば各値に差分を添える */
  previous: MatchStats | null
  /** 表示中の結果が、現在のパラメータと異なる設定で得たものか */
  stale: boolean
  isPending: boolean
}

/** 時間内訳に出す段階 (SPEC 4.4 の elapsed_ms のキー)。`total` は各段階の合計とは別に表示する。 */
const STAGES: { key: string; label: string }[] = [
  { key: 'preprocess', label: 'パッチ抽出' },
  { key: 'features', label: '特徴' },
  { key: 'solve', label: 'Sinkhorn 求解' },
  { key: 'divergence', label: 'ダイバージェンス' },
  { key: 'plan', label: '計画の要約' },
  { key: 'hard', label: 'ハード割当' },
]

export function StatsPanel({ result, previous, stale, isPending }: Props) {
  return (
    <section aria-label="統計" className="min-w-0 rounded border border-slate-800 bg-slate-900/60 p-4">
      <h2 className="mb-3 text-sm font-semibold text-slate-200">統計</h2>
      {result ? (
        <Body result={result} previous={previous} stale={stale} isPending={isPending} />
      ) : (
        <p className="text-xs text-slate-500">
          {isPending ? '計算中…' : '実行すると、ダイバージェンス・ε・時間内訳・警告がここに表示されます。'}
        </p>
      )}
    </section>
  )
}

function Body({ result, previous, stale, isPending }: Props & { result: MatchResponse }) {
  const { stats, warnings } = result
  const ms = (s: MatchStats, key: string) => s.elapsed_ms[key] ?? 0
  const total = ms(stats, 'total')
  const previousRowError = previous?.row_mass_error != null ? previous.row_mass_error * 100 : null

  return (
    <div className="space-y-4 text-xs">
      {stale && !isPending && (
        <p className="rounded border border-sky-900 bg-sky-950/60 px-3 py-2 text-sky-300" role="status">
          表示中の結果は、現在のパラメータとは別の設定で計算したものです。「実行」で更新できます。
        </p>
      )}

      <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1.5">
        <Row label="パッチ数">
          A: {stats.n_patches_a} / B: {stats.n_patches_b} · 特徴 {stats.feature_dim} 次元
        </Row>
        <Row label="ε (= blur²)">{formatNumber(stats.eps)}</Row>
        <Row label="ダイバージェンス">
          {stats.sinkhorn_divergence === null ? (
            <span className="text-slate-500">未計算</span>
          ) : (
            <>
              {formatNumber(stats.sinkhorn_divergence, 4)}
              <Delta current={stats.sinkhorn_divergence} previous={previous?.sinkhorn_divergence} digits={2} />
            </>
          )}
        </Row>
        <Row label="反復">
          {stats.iterations} 回 · {stats.converged ? '収束' : '上限で停止'}
          <Delta current={stats.iterations} previous={previous?.iterations} />
        </Row>
        <Row label="行質量の誤差">
          {stats.row_mass_error === null ? (
            <span className="text-slate-500">— (unbalanced)</span>
          ) : (
            <>
              {(stats.row_mass_error * 100).toFixed(1)}%
              <Delta current={stats.row_mass_error * 100} previous={previousRowError} unit="pt" digits={2} />
            </>
          )}
        </Row>
        <Row label="ハード / top-1 一致">{(stats.hard_agreement * 100).toFixed(1)}%</Row>
        <Row label="バックエンド">
          {stats.backend} ({stats.device})
        </Row>
      </dl>

      <div>
        <h3 className="mb-1.5 font-semibold text-slate-300">時間内訳</h3>
        <ul className="space-y-1">
          {STAGES.map(({ key, label }) => {
            const value = ms(stats, key)
            return (
              <li key={key} className="grid grid-cols-[7.5rem_1fr_auto] items-center gap-2">
                <span className="truncate text-slate-400">{label}</span>
                <span className="h-1.5 overflow-hidden rounded-full bg-slate-800" aria-hidden>
                  <span
                    className="block h-full rounded-full bg-sky-500"
                    style={{ width: `${total > 0 ? Math.min(100, (value / total) * 100) : 0}%` }}
                  />
                </span>
                <span className="text-right tabular-nums text-slate-300">
                  {Math.round(value)} ms
                  <Delta
                    current={value}
                    previous={previous ? ms(previous, key) : undefined}
                    unit="ms"
                    digits={3}
                    round
                  />
                </span>
              </li>
            )
          })}
          <li className="grid grid-cols-[7.5rem_1fr_auto] items-center gap-2 border-t border-slate-800 pt-1 font-semibold text-slate-200">
            <span>合計</span>
            <span />
            <span className="text-right tabular-nums">
              {Math.round(total)} ms
              <Delta
                current={total}
                previous={previous ? ms(previous, 'total') : undefined}
                unit="ms"
                digits={3}
                round
              />
            </span>
          </li>
        </ul>
      </div>

      {warnings.length > 0 && (
        <div className="rounded border border-amber-800 bg-amber-950/60 px-3 py-2 text-amber-300" role="status">
          <h3 className="mb-1 font-semibold">警告</h3>
          <ul className="space-y-1">
            {warnings.map((w, k) => (
              <li key={k} title={w.message}>
                ⚠ {describeWarning(w)}
              </li>
            ))}
          </ul>
        </div>
      )}

      {previous && <p className="text-slate-500">括弧内の灰色の数値は、直前の実行 (同じ画像ペア) との差です。</p>}
    </div>
  )
}

function Row({ label, children }: { label: string; children: ReactNode }) {
  return (
    <>
      <dt className="text-slate-400">{label}</dt>
      <dd className="min-w-0 text-slate-200">{children}</dd>
    </>
  )
}

/** 直前の実行との差 (符号つき、灰色)。前回が無い / 比べられないときは何も出さない。 */
function Delta({
  current,
  previous,
  unit = '',
  digits = 3,
  round = false,
}: {
  current: number
  previous: number | null | undefined
  unit?: string
  digits?: number
  /** 整数に丸めた差にする (ms 表示用) */
  round?: boolean
}) {
  if (previous === null || previous === undefined) return null
  const diff = round ? Math.round(current) - Math.round(previous) : current - previous
  return (
    <span className="ml-1.5 text-slate-500">
      ({formatDelta(diff, digits)}
      {unit && ` ${unit}`})
    </span>
  )
}
