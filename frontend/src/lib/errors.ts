import { ApiError } from '../api/client.ts'

export interface ErrorText {
  /** 一行の見出し (日本語) */
  title: string
  /** サーバが返したメッセージ (英語) や補足。無ければ null */
  detail: string | null
  hint: string | null
}

const TITLES: Partial<Record<ApiError['code'], string>> = {
  IMAGE_TOO_LARGE: '画像が大きすぎます',
  UNSUPPORTED_FORMAT: '対応していない画像形式です (PNG / JPEG / WebP)',
  IMAGE_NOT_FOUND: '画像が見つかりません (保存期限切れの可能性があります)',
  TOO_MANY_PATCHES: 'パッチ数が上限を超えています',
  INVALID_PARAMS: 'パラメータが不正です',
  SOLVER_FAILED: '最適輸送の計算に失敗しました',
  NETWORK_ERROR: 'サーバに接続できません',
}

/** 例外を UI に出す文言へ整形する。サーバのメッセージは握りつぶさず detail に残す。 */
export function describeError(e: unknown): ErrorText {
  if (e instanceof ApiError) {
    const title = TITLES[e.code]
    if (title) return { title, detail: e.code === 'NETWORK_ERROR' ? null : e.message, hint: e.hint }
    return { title: 'エラーが発生しました', detail: e.message, hint: e.hint }
  }
  return { title: 'エラーが発生しました', detail: e instanceof Error ? e.message : String(e), hint: null }
}

export function isAbortError(e: unknown): boolean {
  return e instanceof DOMException && e.name === 'AbortError'
}
