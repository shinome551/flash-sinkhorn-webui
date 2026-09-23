import { describe, expect, it } from 'vitest'
import type { HealthResponse } from '../api/types.ts'
import { computeGrid } from './grid.ts'
import { featureUnavailable, flashUnavailable, runBlocker, suggestPatchSize, type PreflightInput } from './preflight.ts'

function health(
  cuda: boolean,
  maxPatches: number,
  defaultBackend = cuda ? 'flash' : 'dense',
  featureTypes: HealthResponse['feature_types'] = ['pca', 'raw', 'color', 'dinov2'],
): HealthResponse {
  return {
    status: 'ok',
    device: cuda ? 'cuda:0' : 'cpu',
    cuda_available: cuda,
    gpu_name: null,
    torch_version: '',
    flash_sinkhorn_version: cuda ? '0.3.3.post1' : null,
    default_backend: defaultBackend,
    feature_types: featureTypes,
    limits: { max_patches: maxPatches, max_upload_bytes: 1 },
  }
}

// 512×384 / patch 16 → 32×24 = 768 パッチ
const grid = computeGrid(512, 384, 16)
const ready: PreflightInput = {
  hasImageA: true,
  hasImageB: true,
  paramsValid: true,
  gridA: grid,
  gridB: grid,
  health: health(true, 4096),
  backend: '',
  featureType: 'pca',
}

describe('runBlocker', () => {
  it('条件がそろえば undefined', () => {
    expect(runBlocker(ready)).toBeUndefined()
  })

  it('画像が片方でも無ければ止める', () => {
    expect(runBlocker({ ...ready, hasImageB: false })).toContain('画像 A と画像 B')
  })

  it('パラメータが不正なら止める', () => {
    expect(runBlocker({ ...ready, paramsValid: false })).toContain('不正な値')
  })

  it('CUDA が無いのに flash を選んでいたら止める', () => {
    expect(runBlocker({ ...ready, health: health(false, 1024), backend: 'flash' })).toContain('flash')
    expect(runBlocker({ ...ready, health: health(false, 1024), backend: 'dense' })).toBeUndefined()
  })

  it('サーバに無い特徴 (extra deep 未導入の dinov2) を選んでいたら止める', () => {
    const noDeep = health(true, 4096, 'flash', ['pca', 'raw', 'color'])
    expect(runBlocker({ ...ready, health: noDeep, featureType: 'dinov2' })).toContain('--extra deep')
    expect(runBlocker({ ...ready, health: noDeep, featureType: 'pca' })).toBeUndefined()
    expect(runBlocker({ ...ready, featureType: 'dinov2' })).toBeUndefined()
  })

  it('パッチ数が上限を超えたら、収まる patch size を添えて止める', () => {
    const msg = runBlocker({ ...ready, health: health(false, 512) })
    expect(msg).toContain('上限 (512)')
    expect(msg).toContain('patch size を 20 以上')
  })

  it('health が無い (サーバ停止中) ときは上限・バックエンドを判定しない', () => {
    expect(runBlocker({ ...ready, health: undefined, backend: 'flash' })).toBeUndefined()
  })
})

describe('featureUnavailable', () => {
  it('health.feature_types に無い種類だけ true', () => {
    const noDeep = health(true, 4096, 'flash', ['pca', 'raw', 'color'])
    expect(featureUnavailable(noDeep, 'dinov2')).toBe(true)
    expect(featureUnavailable(noDeep, 'raw')).toBe(false)
  })
})

describe('flashUnavailable', () => {
  it('backend 未指定ならサーバの既定値で判定する', () => {
    expect(flashUnavailable(health(false, 1024, 'flash'), '')).toBe(true)
    expect(flashUnavailable(health(false, 1024, 'dense'), '')).toBe(false)
    expect(flashUnavailable(health(true, 4096), 'flash')).toBe(false)
  })

  it('CUDA があっても flash-sinkhorn が無ければ true', () => {
    const noFlash = { ...health(true, 4096, 'dense'), flash_sinkhorn_version: null }
    expect(flashUnavailable(noFlash, 'flash')).toBe(true)
    expect(flashUnavailable(noFlash, '')).toBe(false)
  })
})

describe('suggestPatchSize', () => {
  it('両画像が上限に収まる最小の size (サーバの hint と同じ)', () => {
    // 20px: 25×19 = 475 ≤ 512、19px: 26×20 = 520 > 512
    expect(suggestPatchSize(grid, grid, 512)).toBe(20)
  })

  it('stride < size で超過したときは、同じ size (stride = size) から探す', () => {
    // 512×384 / size 16・stride 8 → 63×47 = 2961 > 1024。16 (stride 16) なら 32×24 = 768 で収まる
    const overlapping = computeGrid(512, 384, 16, 8)
    expect(suggestPatchSize(overlapping, overlapping, 1024)).toBe(16)
  })

  it('どの size でも収まらなければ null', () => {
    expect(suggestPatchSize(grid, grid, 0)).toBeNull()
  })
})
