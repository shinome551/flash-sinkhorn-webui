import { expect, test, type Page } from '@playwright/test'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const SAMPLES = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../assets/samples')

/** HoverInfo の「#i (行 r, 列 c)」を読む。1 つ目が hover 中のパッチ、2 つ目が top-1 の対応先。 */
async function hoveredAndTop1(page: Page) {
  const text = await page.getByText(/確信度 [\d.]+ · エントロピー/).textContent()
  const cells = [...(text ?? '').matchAll(/#\d+ \(行 (\d+), 列 (\d+)\)/g)].map((m) => ({
    row: Number(m[1]),
    col: Number(m[2]),
  }))
  expect(cells.length).toBeGreaterThanOrEqual(2)
  return { hovered: cells[0], top1: cells[1] }
}

test('アップロード → 実行 → ホバーで対応先をハイライト', async ({ page }) => {
  await page.goto('/')
  await expect(page.getByTestId('run-blocker')).toContainText('画像 A と画像 B を選択してください')
  await expect(page.getByRole('banner').getByRole('button', { name: '実行' })).toBeDisabled()

  // 平行移動のペア: A の (x, y) の内容は B の (x + 32, y + 16) にある = パッチ 2 列・1 行ずれ
  await page.getByLabel(/画像 A をドロップ/).setInputFiles(path.join(SAMPLES, 'cat_a.jpg'))
  await page.getByLabel(/画像 B をドロップ/).setInputFiles(path.join(SAMPLES, 'cat_b.jpg'))
  await expect(page.getByText('400×272')).toHaveCount(2)
  await expect(page.getByTestId('run-blocker')).toHaveCount(0)

  // パラメータパネル側のボタンでも実行できる
  await page.getByRole('region', { name: 'パラメータ' }).getByRole('button', { name: '実行' }).click()
  const stats = page.getByRole('region', { name: '統計' })
  await expect(stats.getByText('合計')).toBeVisible({ timeout: 30_000 })
  await expect(stats).toContainText('A: 425 / B: 425')

  // 画像 A の中央付近 (行 9, 列 9 あたり) にカーソルを置く
  const picker = page.getByRole('group', { name: /画像 A のパッチ選択/ })
  const box = (await picker.boundingBox())!
  const hoverCenter = () => page.mouse.move(box.x + box.width * 0.4, box.y + box.height * 0.5)
  await hoverCenter()

  const overlay = page.locator('svg[aria-hidden]').filter({ has: page.locator('line') })
  await expect(overlay.locator('line').first()).toBeAttached()
  await expect(overlay.locator('rect').first()).toBeAttached()

  const { hovered, top1 } = await hoveredAndTop1(page)
  expect(top1).toEqual({ row: hovered.row + 1, col: hovered.col + 2 })

  // ハード割当に切り替えても (再実行なし) 対応先は top-1 と同じで、HoverInfo の表記が変わる
  await page.getByRole('radio', { name: 'ハード' }).check({ force: true })
  await hoverCenter() // ラジオのクリックでカーソルが A から外れるので戻す
  await expect(page.getByText(/ハード割当: #\d+/)).toBeVisible()
  expect(await hoveredAndTop1(page)).toEqual({ hovered, top1 })
  await expect(stats).toContainText('ハード / top-1 一致')
  await page.getByRole('radio', { name: 'ソフト' }).check({ force: true })

  // カーソルを外すとハイライトが消える
  await page.mouse.move(0, 0)
  await expect(page.getByText(/A の画像にカーソルを置く/)).toBeVisible()
})
