import { useCallback, useLayoutEffect, useState, type RefObject } from 'react'
import type { BoxGeometry } from '../lib/overlay.ts'

interface Geometry {
  a: BoxGeometry | null
  b: BoxGeometry | null
}

const EPSILON = 0.01

function same(p: BoxGeometry | null, q: BoxGeometry | null): boolean {
  if (p === null || q === null) return p === q
  return Math.abs(p.x - q.x) < EPSILON && Math.abs(p.y - q.y) < EPSILON && Math.abs(p.scale - q.scale) < 1e-6
}

/** box (画像を描く要素) の、ステージ左上を原点とした位置と表示倍率。まだ寸法が無ければ null。 */
function measure(stage: HTMLElement | null, box: HTMLElement | null, imageWidth: number | undefined) {
  if (!stage || !box || !imageWidth) return null
  const s = stage.getBoundingClientRect()
  const b = box.getBoundingClientRect()
  if (b.width <= 0) return null
  return { x: b.left - s.left, y: b.top - s.top, scale: b.width / imageWidth }
}

/**
 * 2 枚の画像 box をステージ座標に変換するための位置・倍率を追跡する。
 * レイアウト (2 列 ⇄ 縦積み)・ウィンドウ・画像の表示幅が変わるたびに再測定する。
 * `remeasureKey` が変わったとき (画像の差し替えで box が作り直されるとき) も測り直す。
 */
export function useOverlayGeometry(
  stageRef: RefObject<HTMLElement | null>,
  boxARef: RefObject<HTMLElement | null>,
  boxBRef: RefObject<HTMLElement | null>,
  imageWidthA: number | undefined,
  imageWidthB: number | undefined,
  remeasureKey: string,
): Geometry {
  const [geometry, setGeometry] = useState<Geometry>({ a: null, b: null })

  const update = useCallback(() => {
    const a = measure(stageRef.current, boxARef.current, imageWidthA)
    const b = measure(stageRef.current, boxBRef.current, imageWidthB)
    setGeometry((prev) => (same(prev.a, a) && same(prev.b, b) ? prev : { a, b }))
  }, [stageRef, boxARef, boxBRef, imageWidthA, imageWidthB])

  useLayoutEffect(() => {
    update()
    const observer = new ResizeObserver(update)
    for (const ref of [stageRef, boxARef, boxBRef]) {
      if (ref.current) observer.observe(ref.current)
    }
    window.addEventListener('resize', update)
    return () => {
      observer.disconnect()
      window.removeEventListener('resize', update)
    }
  }, [update, stageRef, boxARef, boxBRef, remeasureKey])

  return geometry
}
