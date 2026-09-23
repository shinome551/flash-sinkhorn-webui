import { expect, test, type Page } from '@playwright/test'

// エラー系の UI。サーバの状態は /api/* をモックして作る (実バックエンドは画像の登録にだけ使う)。

const cpuHealth = {
  status: 'ok',
  device: 'cpu',
  cuda_available: false,
  gpu_name: null,
  torch_version: '2.11.0',
  flash_sinkhorn_version: '0.3.3',
  default_backend: 'dense',
  feature_types: ['pca', 'raw', 'color'],
  limits: { max_patches: 300, max_upload_bytes: 10 * 1024 * 1024 },
}

async function loadShiftSample(page: Page) {
  await page.getByRole('button', { name: '平行移動' }).click()
  await expect(page.getByText('400×272')).toHaveCount(2)
}

test('サーバ停止中: 案内と再接続', async ({ page }) => {
  let down = true
  await page.route('**/api/health', (route) => (down ? route.abort('connectionrefused') : route.fallback()))
  await page.goto('/')
  await expect(page.getByRole('alert').filter({ hasText: 'バックエンドに接続できません' })).toBeVisible()
  await expect(page.getByText('サーバ停止中')).toBeVisible()

  down = false
  await page.getByRole('button', { name: '再接続' }).click()
  await expect(page.getByText('バックエンドに接続できません')).toHaveCount(0)
})

test('GPU なし: CPU の案内、flash 指定とパッチ数超過で実行を止める', async ({ page }) => {
  await page.route('**/api/health', (route) => route.fulfill({ json: cpuHealth }))
  await page.goto('/')
  await expect(page.getByText('GPU なし (CPU)')).toBeVisible()
  await expect(page.getByText(/CPU の dense バックエンドで計算します/)).toBeVisible()

  await loadShiftSample(page)
  // 400×272 / patch 16 = 425 パッチ > 上限 300。19px なら 21×14 = 294
  const blocker = page.getByTestId('run-blocker')
  await expect(blocker).toContainText('パッチ数が上限 (300) を超えています。patch size を 19 以上')
  await expect(page.getByRole('button', { name: '実行' })).toBeDisabled()

  await page.getByLabel('size [px]').fill('19')
  await expect(blocker).toHaveCount(0)

  await page.getByLabel('backend').selectOption('flash')
  await expect(blocker).toContainText('flash バックエンドは使えません')
  await expect(page.getByText('CUDA か flash-sinkhorn が無いので flash は使えません')).toBeVisible()
  await expect(page.getByRole('button', { name: '実行' })).toBeDisabled()
})

test('実行時のエラー (TOO_MANY_PATCHES) はサーバの hint と一緒に出す', async ({ page }) => {
  await page.goto('/')
  await loadShiftSample(page)
  await page.route('**/api/match', (route) =>
    route.fulfill({
      status: 422,
      json: {
        detail: {
          code: 'TOO_MANY_PATCHES',
          message: 'Too many patches: A=425, B=425 (limit 300 each).',
          hint: 'Use patch size 20 (stride 20) or larger.',
        },
      },
    }),
  )
  await page.getByRole('button', { name: '実行' }).click()
  const alert = page.getByRole('alert').filter({ hasText: 'パッチ数が上限を超えています' })
  await expect(alert).toContainText('Use patch size 20')
})

test('実行中にサーバが落ちたら、停止の案内に切り替える', async ({ page }) => {
  await page.goto('/')
  await loadShiftSample(page)
  await page.route('**/api/match', (route) => route.abort('connectionrefused'))
  await page.route('**/api/health', (route) => route.abort('connectionrefused'))
  await page.getByRole('button', { name: '実行' }).click()
  await expect(page.getByRole('alert').filter({ hasText: 'サーバに接続できません' })).toBeVisible()
  await expect(page.getByText('バックエンドに接続できません')).toBeVisible()
})
