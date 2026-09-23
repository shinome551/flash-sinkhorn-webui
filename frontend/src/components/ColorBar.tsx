import { viridisGradientCss } from '../lib/colormap.ts'
import { formatRange } from '../lib/format.ts'

/** ヒートマップ・矢印の凡例。値の範囲は外れ値を飽和させた分位点 (両端 2%) なので、範囲外の値は端の色になる。 */
export function ColorBar({ label, lo, hi }: { label: string; lo: number; hi: number }) {
  const [loText, hiText] = formatRange(lo, hi)
  return (
    <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-slate-400">
      <span className="text-slate-300">{label}</span>
      <span className="inline-flex items-center gap-1.5">
        <span className="tabular-nums">{loText}</span>
        <span
          className="inline-block h-2.5 w-40 rounded-sm ring-1 ring-slate-700"
          style={{ background: viridisGradientCss() }}
          role="img"
          aria-label={`viridis: ${loText} から ${hiText}`}
        />
        <span className="tabular-nums">{hiText}</span>
      </span>
    </div>
  )
}
