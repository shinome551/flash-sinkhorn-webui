import { useEffect, useRef, type KeyboardEvent, type MouseEvent } from 'react'
import type { GridInfo } from '../api/types.ts'
import { patchAtDisplay } from '../lib/grid.ts'
import { isArrowKey, movePatch } from '../lib/selection.ts'

interface Props {
  /** 画像 A の格子 (マッチ結果のもの) */
  grid: GridInfo
  /** 画像座標→表示座標の倍率 (ImagePanel の displayScale) */
  displayScale: number
  active: number | null
  onActiveChange: (i: number | null) => void
  /** 省略すると固定 (クリック / Enter) を受け付けない (hover / flow モード) */
  onTogglePin?: (i: number) => void
}

/**
 * 画像 A の上に重ねる透明な入力層。マウスの hover / クリックと、矢印キー・Enter・Esc を扱う。
 * 描画はしない (枠や接続線は MatchOverlay)。同じパッチなら state は変わらないので再描画されない。
 */
export function PatchPicker({ grid, displayScale, active, onActiveChange, onTogglePin }: Props) {
  // 矢印キーで再開できるよう、最後に指していたパッチを覚えておく (hover が外れて active が null になっても)。
  const lastRef = useRef<number | null>(null)
  useEffect(() => {
    if (active !== null) lastRef.current = active
  }, [active])

  // PointerEvent は MouseEvent を継承しているので、pointer / click の両方に使える。
  const patchAtPointer = (e: MouseEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect()
    return patchAtDisplay(grid, e.clientX - rect.left, e.clientY - rect.top, displayScale)
  }

  const onKeyDown = (e: KeyboardEvent<HTMLDivElement>) => {
    if (isArrowKey(e.key)) {
      const next = movePatch(grid, active ?? lastRef.current, e.key, e.shiftKey ? 4 : 1)
      if (next !== null) onActiveChange(next)
      e.preventDefault()
    } else if (e.key === 'Enter' || e.key === ' ') {
      if (active !== null) onTogglePin?.(active)
      e.preventDefault()
    } else if (e.key === 'Escape') {
      onActiveChange(null)
    }
  }

  return (
    <div
      className="absolute inset-0 cursor-crosshair rounded outline-offset-2 focus-visible:outline-2 focus-visible:outline-sky-400"
      tabIndex={0}
      role="group"
      aria-label={
        onTogglePin
          ? '画像 A のパッチ選択。矢印キーで移動、Enter で固定、Esc で解除'
          : '画像 A のパッチ選択。矢印キーで移動、Esc で解除'
      }
      onPointerDown={(e) => onActiveChange(patchAtPointer(e))}
      onPointerMove={(e) => onActiveChange(patchAtPointer(e))}
      onPointerLeave={() => onActiveChange(null)}
      onClick={(e) => {
        const i = patchAtPointer(e)
        if (i !== null) onTogglePin?.(i)
      }}
      onKeyDown={onKeyDown}
      onBlur={() => onActiveChange(null)}
    />
  )
}
