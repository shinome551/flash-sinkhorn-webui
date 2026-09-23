// 「実行」前に分かる失敗を、リクエストを送る前に止めて理由を出す。サーバ側の検証 (422) と同じ条件。

import type { BackendChoice, FeatureType, GridInfo, HealthResponse } from '../api/types.ts'
import { computeGrid, patchCount } from './grid.ts'

export interface PreflightInput {
  hasImageA: boolean
  hasImageB: boolean
  paramsValid: boolean
  /** 現在のパッチ設定での格子。画像が無い・設定が不正なら null */
  gridA: GridInfo | null
  gridB: GridInfo | null
  /** 取得できていなければ undefined (サーバ停止中は判定しない) */
  health: HealthResponse | undefined
  /** '' = サーバ設定 */
  backend: '' | BackendChoice
  featureType: FeatureType
}

/** 実行できない理由。実行できるなら undefined。 */
export function runBlocker(input: PreflightInput): string | undefined {
  const { hasImageA, hasImageB, paramsValid, gridA, gridB, health, backend, featureType } = input
  if (!hasImageA || !hasImageB) return '画像 A と画像 B を選択してください (サンプルでも試せます)。'
  if (!paramsValid) return 'パラメータに不正な値があります。'
  if (health && flashUnavailable(health, backend)) {
    return 'このサーバでは flash バックエンドは使えません (CUDA または flash-sinkhorn が無い)。backend を auto か dense にしてください。'
  }
  if (health && featureUnavailable(health, featureType)) {
    return `特徴 ${featureType} はサーバで使えません (バックエンドで \`uv sync --extra deep\` を実行して再起動してください)。`
  }
  const max = health?.limits.max_patches
  if (max !== undefined && gridA && gridB && Math.max(patchCount(gridA), patchCount(gridB)) > max) {
    const size = suggestPatchSize(gridA, gridB, max)
    return (
      `パッチ数が上限 (${max}) を超えています。` +
      (size ? `patch size を ${size} 以上 (stride は「size と同じ」) にしてください。` : '画像を小さくしてください。')
    )
  }
  return undefined
}

/** 選んだ (または既定の) バックエンドが flash なのに、CUDA か flash-sinkhorn が無い。 */
export function flashUnavailable(health: HealthResponse, backend: '' | BackendChoice): boolean {
  const usable = health.cuda_available && health.flash_sinkhorn_version !== null
  return !usable && (backend || health.default_backend) === 'flash'
}

/** 選んだ特徴の種類がサーバに無い (dinov2 で extra `deep` が未導入)。 */
export function featureUnavailable(health: HealthResponse, featureType: FeatureType): boolean {
  return !health.feature_types.includes(featureType)
}

/**
 * 重ならない格子 (stride = size) で両画像が上限に収まる最小の patch size。サーバの hint と同じ探索。
 * 現在の size から探す (stride < size で超過したなら、同じ size を stride = size にすれば収まりうる)。
 */
export function suggestPatchSize(gridA: GridInfo, gridB: GridInfo, max: number): number | null {
  const dims = [gridA, gridB].map((g) => [g.image_width, g.image_height] as const)
  const limit = Math.min(...dims.flat())
  for (let size = gridA.patch_size; size <= limit; size++) {
    if (dims.every(([w, h]) => patchCount(computeGrid(w, h, size)) <= max)) return size
  }
  return null
}
