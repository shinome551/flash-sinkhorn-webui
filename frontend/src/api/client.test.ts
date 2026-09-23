import { describe, expect, it } from 'vitest'
import { ApiError, toApiError } from './client.ts'

describe('toApiError', () => {
  it('サーバのエラー形式 {detail: {code, message, hint}} を保つ', () => {
    const body = JSON.stringify({
      detail: { code: 'TOO_MANY_PATCHES', message: 'too many', hint: 'use patch.size >= 24' },
    })
    const e = toApiError(422, body)
    expect(e).toBeInstanceOf(ApiError)
    expect([e.code, e.message, e.hint, e.status]).toEqual(['TOO_MANY_PATCHES', 'too many', 'use patch.size >= 24', 422])
  })

  it('hint が無ければ null', () => {
    const e = toApiError(404, JSON.stringify({ detail: { code: 'IMAGE_NOT_FOUND', message: 'gone' } }))
    expect(e.hint).toBeNull()
  })

  it('本文の無い 5xx (バックエンド停止中のプロキシ) は NETWORK_ERROR', () => {
    expect(toApiError(502, '').code).toBe('NETWORK_ERROR')
  })

  it('想定外の 4xx は握りつぶさず UNKNOWN にする', () => {
    const e = toApiError(400, '<html>bad</html>')
    expect(e.code).toBe('UNKNOWN')
    expect(e.message).toContain('400')
  })

  it('FastAPI 既定の {detail: "文字列"} は形式外として扱う', () => {
    expect(toApiError(404, JSON.stringify({ detail: 'Not Found' })).code).toBe('UNKNOWN')
  })
})
