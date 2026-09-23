// バックエンドのスキーマ (backend/app/schemas/) に対応する型。食い違ったら Pydantic 側を正とする。

// ---- エラー (SPEC 4.5) ----

export type ApiErrorCode =
  | 'IMAGE_TOO_LARGE'
  | 'UNSUPPORTED_FORMAT'
  | 'IMAGE_NOT_FOUND'
  | 'TOO_MANY_PATCHES'
  | 'INVALID_PARAMS'
  | 'SOLVER_FAILED'
  | 'HTTP_ERROR'
  | 'INTERNAL_ERROR'
  // 以下はクライアント側で付与する (サーバは返さない)
  | 'NETWORK_ERROR' // サーバに到達できない (停止中・プロキシ失敗を含む)
  | 'UNKNOWN' // 想定した形式で読めなかった応答

export interface ErrorDetail {
  code: ApiErrorCode
  message: string
  hint: string | null
}

// ---- GET /api/health (schemas/health.py) ----

export interface HealthResponse {
  status: string
  device: string
  cuda_available: boolean
  gpu_name: string | null
  torch_version: string
  flash_sinkhorn_version: string | null
  default_backend: string
  /** 利用できる feature.type (dinov2 はバックエンドに extra `deep` があるときのみ) */
  feature_types: FeatureType[]
  limits: { max_patches: number; max_upload_bytes: number }
}

// ---- POST /api/images (schemas/image.py) ----

export interface ImageUploadResponse {
  image_id: string
  /** 前処理後の寸法。GET /api/images/{id} が返す PNG の寸法と一致する。 */
  width: number
  height: number
  /** EXIF 回転後・縮小前の寸法 */
  original_width: number
  original_height: number
  url: string
}

// ---- POST /api/match (schemas/match.py) ----

export type FeatureType = 'raw' | 'pca' | 'color' | 'dinov2'
export type NormalizeMode = 'none' | 'zscore' | 'l2'
export type BackendChoice = 'auto' | 'flash' | 'dense'

export interface PatchRequest {
  size?: number
  /** 省略時は size */
  stride?: number | null
}

export interface FeatureRequest {
  type?: FeatureType
  pca_dim?: number
  normalize?: NormalizeMode
  position_weight?: number
}

export interface OTRequest {
  blur?: number
  scaling?: number
  half_cost?: boolean
  reach_x?: number | null
  reach_y?: number | null
  /** ε 相対。null で打ち切りなし */
  threshold?: number | null
  inner_iterations?: number
  /** 省略時はサーバ設定 */
  backend?: BackendChoice | null
}

export interface OutputRequest {
  top_k?: number
  min_weight?: number
  compute_divergence?: boolean
}

/** 未知のキーはサーバが拒否する (extra="forbid")。 */
export interface MatchRequest {
  image_a: string
  image_b: string
  patch?: PatchRequest
  feature?: FeatureRequest
  ot?: OTRequest
  output?: OutputRequest
}

export interface GridInfo {
  rows: number
  cols: number
  patch_size: number
  stride: number
  image_width: number
  image_height: number
}

export interface Target {
  j: number
  /** 条件付き確率 p̃_ij */
  weight: number
  cost: number
}

/** c-transform の argmin j* = argmin_j [C_ij - g_j] (SPEC 3.6)。b が一様ならソフトの top-1 と一致する */
export interface HardTarget {
  j: number
  /** C_ij* */
  cost: number
  /** center_B(j*) - center_A(i)。ピクセル単位 (dx, dy) */
  displacement: [number, number]
}

export interface PatchMatch {
  i: number
  confidence: number
  entropy: number
  /** a_i で割った値 (balanced なら ≈ 1) */
  row_mass: number
  /** ピクセル単位 (dx, dy) */
  displacement: [number, number]
  /** weight 降順。0 件もありうる */
  targets: Target[]
  hard: HardTarget
}

export interface MatchStats {
  n_patches_a: number
  n_patches_b: number
  feature_dim: number
  eps: number
  sinkhorn_divergence: number | null
  backend: string
  device: string
  iterations: number
  converged: boolean
  /** balanced のときの max|row_mass/a - 1|。unbalanced は null */
  row_mass_error: number | null
  /** hard.j がソフトの top-1 と一致した割合 (b が一様なら ≈ 1) */
  hard_agreement: number
  /** preprocess / features / solve / divergence / plan / hard / total */
  elapsed_ms: Record<string, number>
}

/** code は backend/app/services/notices.py の WarningCode。文言は lib/warnings.ts */
export interface MatchWarning {
  code: string
  /** 英語の説明 (未知のコードのときに表示する) */
  message: string
  params: Record<string, number | boolean>
}

export interface MatchResponse {
  grid_a: GridInfo
  grid_b: GridInfo
  /** A 側パッチの i 昇順 (i = row * cols + col) */
  matches: PatchMatch[]
  /** B 側の受け取り質量を b_j で割った値 (1.0 が期待値) */
  col_mass: number[]
  /** ハード割当での受け取り量 (選ばれた A パッチの質量 / b_j)。1.0 = ちょうど 1 個 */
  hard_col_mass: number[]
  stats: MatchStats
  /** 握りつぶさず StatsPanel に出す */
  warnings: MatchWarning[]
}

// ---- GET /api/samples, POST /api/samples/{id} (schemas/sample.py) ----

export interface SampleInfo {
  id: string
  title: string
  description: string
}

export interface SamplePairResponse {
  a: ImageUploadResponse
  b: ImageUploadResponse
}
