import { memo, useId } from 'react'
import type { Flow } from '../lib/flow.ts'
import { HALO_COLOR, type Overlay } from '../lib/overlay.ts'

/**
 * ステージ全体 (画像 A・B とその間) に重ねる SVG。座標は `buildOverlay` / `buildFlow` が作ったステージ内 CSS px。
 * マウスは通さない (入力は画像 A 上の PatchPicker が受ける)。
 */
export const MatchOverlay = memo(function MatchOverlay({ overlay, flow }: { overlay: Overlay; flow: Flow | null }) {
  return (
    <svg className="pointer-events-none absolute inset-0 z-10 h-full w-full" aria-hidden>
      {flow && <FlowArrows flow={flow} />}
      {[...overlay.rectsA, ...overlay.rectsB].map((r) => (
        <rect
          key={r.key}
          x={r.x}
          y={r.y}
          width={r.width}
          height={r.height}
          fill={r.fill}
          fillOpacity={r.fillOpacity}
          stroke={r.stroke}
          strokeOpacity={r.strokeOpacity}
          strokeWidth={r.strokeWidth}
        />
      ))}
      {overlay.lines.map((l) => (
        <g key={l.key} strokeLinecap="round">
          <line
            x1={l.x1}
            y1={l.y1}
            x2={l.x2}
            y2={l.y2}
            stroke={HALO_COLOR}
            strokeOpacity={l.opacity}
            strokeWidth={l.width + 2}
          />
          <line
            x1={l.x1}
            y1={l.y1}
            x2={l.x2}
            y2={l.y2}
            stroke={l.color}
            strokeOpacity={l.opacity}
            strokeWidth={l.width}
          />
          <circle
            cx={l.x2}
            cy={l.y2}
            r={l.width / 2 + 1.5}
            fill={l.color}
            fillOpacity={l.opacity}
            stroke={HALO_COLOR}
            strokeWidth={1}
          />
        </g>
      ))}
    </svg>
  )
})

/** flow の矢印。hover / pin の再描画に巻き込まれないよう分けている (矢印は多いと数千本になる)。 */
const FlowArrows = memo(function FlowArrows({ flow }: { flow: Flow }) {
  const clipId = useId()
  return (
    <g clipPath={`url(#${clipId})`} strokeLinecap="round" strokeLinejoin="round">
      <clipPath id={clipId}>
        <rect x={flow.clip.x} y={flow.clip.y} width={flow.clip.width} height={flow.clip.height} />
      </clipPath>
      {flow.arrows.map((f) =>
        f.head === null ? (
          <circle key={f.key} cx={f.x1} cy={f.y1} r={1.5} fill={f.color} opacity={f.opacity} />
        ) : (
          <g key={f.key} opacity={f.opacity}>
            <line x1={f.x1} y1={f.y1} x2={f.x2} y2={f.y2} stroke={HALO_COLOR} strokeWidth={3.5} />
            <line x1={f.x1} y1={f.y1} x2={f.x2} y2={f.y2} stroke={f.color} strokeWidth={1.5} />
            <polygon
              points={f.head.map((p) => `${p.x},${p.y}`).join(' ')}
              fill={f.color}
              stroke={HALO_COLOR}
              strokeWidth={1}
            />
          </g>
        ),
      )}
    </g>
  )
})
