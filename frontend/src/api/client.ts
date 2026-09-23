import type {
  ApiErrorCode,
  ErrorDetail,
  HealthResponse,
  ImageUploadResponse,
  MatchRequest,
  MatchResponse,
  SampleInfo,
  SamplePairResponse,
} from './types.ts'

// dev では Vite の server.proxy が /api を FastAPI へ転送する。
const API_BASE = '/api'

export class ApiError extends Error {
  readonly code: ApiErrorCode
  readonly hint: string | null
  /** HTTP ステータス。サーバに到達できなかったときは 0 */
  readonly status: number

  constructor(code: ApiErrorCode, message: string, hint: string | null, status: number) {
    super(message)
    this.name = 'ApiError'
    this.code = code
    this.hint = hint
    this.status = status
  }
}

const NETWORK_ERROR_MESSAGE = 'Cannot reach the server.'
const NETWORK_ERROR_HINT = 'Check that the backend is running (uvicorn app.main:app --port 8000).'

function isErrorDetail(v: unknown): v is ErrorDetail {
  return (
    typeof v === 'object' &&
    v !== null &&
    typeof (v as ErrorDetail).code === 'string' &&
    typeof (v as ErrorDetail).message === 'string'
  )
}

/** `{"detail": {code, message, hint}}` を ApiError に整形する。形式外の応答も握りつぶさず ApiError にする。 */
export function toApiError(status: number, bodyText: string): ApiError {
  let detail: unknown
  try {
    detail = (JSON.parse(bodyText) as { detail?: unknown }).detail
  } catch {
    // 本文が JSON でない
  }
  if (isErrorDetail(detail)) {
    return new ApiError(detail.code, detail.message, detail.hint ?? null, status)
  }
  // バックエンド停止中の Vite プロキシは、本文の無い 5xx を返す。
  if (status >= 500) return new ApiError('NETWORK_ERROR', NETWORK_ERROR_MESSAGE, NETWORK_ERROR_HINT, status)
  return new ApiError('UNKNOWN', `Unexpected response (HTTP ${status}).`, null, status)
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response
  try {
    res = await fetch(API_BASE + path, init)
  } catch (e) {
    if (e instanceof DOMException && e.name === 'AbortError') throw e
    throw new ApiError('NETWORK_ERROR', NETWORK_ERROR_MESSAGE, NETWORK_ERROR_HINT, 0)
  }
  const text = await res.text()
  if (!res.ok) throw toApiError(res.status, text)
  try {
    return JSON.parse(text) as T
  } catch {
    throw new ApiError('UNKNOWN', 'The server returned malformed JSON.', null, res.status)
  }
}

export function getHealth(signal?: AbortSignal): Promise<HealthResponse> {
  return request<HealthResponse>('/health', { signal })
}

export function postMatch(req: MatchRequest, signal?: AbortSignal): Promise<MatchResponse> {
  return request<MatchResponse>('/match', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
    signal,
  })
}

export function listSamples(signal?: AbortSignal): Promise<SampleInfo[]> {
  return request<SampleInfo[]>('/samples', { signal })
}

/** 同梱サンプルの 2 枚をサーバの画像ストアに登録する (アップロードと同じ前処理)。 */
export function loadSample(id: string, signal?: AbortSignal): Promise<SamplePairResponse> {
  return request<SamplePairResponse>(`/samples/${encodeURIComponent(id)}`, { method: 'POST', signal })
}

/**
 * 画像をアップロードする。fetch にはアップロード進捗が無いので XMLHttpRequest を使う。
 * `signal` で中断すると AbortError (DOMException) で reject する。
 */
export function uploadImage(
  file: File,
  onProgress?: (fraction: number) => void,
  signal?: AbortSignal,
): Promise<ImageUploadResponse> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(new DOMException('Aborted', 'AbortError'))
      return
    }
    const xhr = new XMLHttpRequest()
    xhr.open('POST', `${API_BASE}/images`)
    xhr.responseType = 'text'

    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) onProgress?.(e.loaded / e.total)
    }
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          resolve(JSON.parse(xhr.responseText) as ImageUploadResponse)
        } catch {
          reject(new ApiError('UNKNOWN', 'The server returned malformed JSON.', null, xhr.status))
        }
        return
      }
      reject(toApiError(xhr.status, xhr.responseText))
    }
    xhr.onerror = () => reject(new ApiError('NETWORK_ERROR', NETWORK_ERROR_MESSAGE, NETWORK_ERROR_HINT, 0))
    xhr.onabort = () => reject(new DOMException('Aborted', 'AbortError'))
    signal?.addEventListener('abort', () => xhr.abort(), { once: true })

    const form = new FormData()
    form.append('file', file)
    xhr.send(form)
  })
}

/**
 * 画像がまだサーバにあるか (TTL 切れの判定用)。GET の本文は読まずに捨てる。
 * サーバに到達できないときは判定できないので true (= 期限切れとは見なさない)。
 */
export async function imageExists(imageId: string, signal?: AbortSignal): Promise<boolean> {
  try {
    const res = await fetch(imageUrl(imageId), { signal })
    void res.body?.cancel()
    return res.status !== 404
  } catch (e) {
    if (e instanceof DOMException && e.name === 'AbortError') throw e
    return true
  }
}

/** 前処理後の PNG の URL。upload の `url` はこれと同じ形式。 */
export function imageUrl(imageId: string): string {
  return `${API_BASE}/images/${encodeURIComponent(imageId)}`
}
