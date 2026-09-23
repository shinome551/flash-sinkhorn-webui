# flash-sinkhorn-webui 仕様書

最終更新: 2026-09-21 / ステータス: Phase 7 まで実装済み (バックエンド完成、フロントは可視化の全モードとパラメータ UI まで)

## 1. 概要

2枚の画像をアップロードし、それぞれをパッチ（格子状のタイル）に分割して、
パッチ同士の対応関係を **エントロピー正則化最適輸送 (Sinkhorn OT)** で求め、
結果をブラウザ上でインタラクティブに可視化する Web アプリケーション。

OT ソルバには [flash-sinkhorn](https://pypi.org/project/flash-sinkhorn/)
(PyTorch + Triton, `n×m` コスト行列を materialize しない streaming 実装) を用いる。

- フロントエンド: React (Vite + TypeScript)
- バックエンド: FastAPI (Python 3.12)
- 計算: CUDA GPU (Triton カーネル)。GPU 非搭載環境向けに CPU リファレンス実装も持つ。

### 1.1 ユースケース

1. 同一シーンの2視点画像 → 視差・対応の概観
2. 前後フレーム → パッチレベルのフロー（疎なオプティカルフロー的表示）
3. スタイル/テクスチャの異なる2画像 → どの領域がどの領域に輸送されるか
4. OT のパラメータ (blur, debias, unbalanced 等) が対応にどう効くかの教育的デモ

### 1.2 非目標 (MVP スコープ外)

- 特徴点検出 (SIFT/SuperPoint 等) ベースのマッチング
- 深層特徴 (DINOv2/CLIP) — Phase 9 のオプション拡張として扱う
- 認証・マルチユーザ・永続DB
- 3枚以上の同時マッチング、動画

## 2. 全体アーキテクチャ

```
[Browser]
  React SPA (Vite)
    ├─ 画像2枚アップロード (drag&drop)
    ├─ パラメータ設定パネル
    ├─ Canvas/SVG によるマッチ可視化
    └─ 統計・エクスポート
        │  JSON over HTTP (fetch)
        ▼
[FastAPI]  uvicorn
    ├─ POST /api/images        画像登録 (一時ストア)
    ├─ GET  /api/images/{id}   画像配信
    ├─ POST /api/match         マッチング実行 (同期)
    └─ GET  /api/health        GPU/バージョン情報
        │
        ▼
[Matching core]  (純粋な Python/PyTorch, FastAPI 非依存)
    画像 → パッチ抽出 → 特徴ベクトル化 → OT ソルバ → 輸送計画の要約
        │
        ▼
[OT backend]  Protocol による抽象化
    ├─ FlashSinkhornBackend (flash_sinkhorn.SamplesLoss, CUDA 必須)
    └─ DenseTorchBackend    (log-domain 素朴実装, CPU/GPU, 検証・フォールバック用)
```

**設計方針**: マッチングコアは FastAPI に依存しない純粋関数群として実装する。
CLI やテストから直接呼べることで、Web 層を経由せずアルゴリズムを検証できる。

## 3. アルゴリズム仕様

### 3.1 前処理

1. アップロード画像を RGB に変換 (EXIF 回転を適用、アルファは白背景に合成)。
2. 長辺が `resize_max` (既定 512px) 以下になるようアスペクト比維持で縮小。
3. 幅・高さを `stride` の倍数に切り詰める (端数パッチを作らない。右端・下端を削る)。
   アップロード時点では `/api/match` の `stride` が未定のため、基準は設定
   `preprocess_stride` (既定 16)。これと異なる `stride` を指定した場合、
   格子に収まらない端は `unfold` と同様に無視される。

### 3.2 パッチ抽出

- パッチサイズ `patch_size` (既定 16)、ストライド `stride` (既定 = patch_size)。
- 格子状に `rows × cols` 個のパッチを取得 (`torch.nn.functional.unfold`)。
- パッチ `i` の代表座標 = パッチ中心のピクセル座標 `(cx_i, cy_i)`。
- パッチ数 `N = rows * cols`。N の上限は `MAX_PATCHES` (既定 4096)。
  超過時は 422 で拒否し、推奨パラメータをメッセージに含める。

### 3.3 特徴ベクトル化

コストは **二乗ユークリッド距離のみ** (flash-sinkhorn は p=2 固定) なので、
「何を対応とみなすか」は特徴設計で表現する。

| `feature.type` | 次元 d | 内容 |
|---|---|---|
| `raw` | `3 * patch_size²` | パッチ画素をそのまま平坦化 ([0,1] スケール) |
| `pca` (既定) | `pca_dim` (既定 64) | `raw` を両画像結合で PCA 射影 (`torch.pca_lowrank`) |
| `color` | 3 | パッチ平均 RGB (軽量・デモ用) |
| `dinov2` | 384 | DINOv2 ViT-S/14 (`timm` の `vit_small_patch14_dinov2.lvd142m`) のパッチトークン。optional extra `deep` が必要で、未導入なら `INVALID_PARAMS` (422)、重みの取得失敗は `INTERNAL_ERROR` (503)。`pca_dim` は使わない |

`dinov2` はトークン間隔を stride に合わせる: 格子が覆う領域 `[0, cols·stride) × [0, rows·stride)` を
`(cols·14, rows·14)` にリサイズ (はみ出す分は端を複製) してモデルに通す。`patch_size == stride` なら
トークン `(r, c)` がパッチ `i = r·cols + c` にそのまま対応し、そうでなければトークンマップをパッチ中心で双線形補間する。

後処理オプション:

- `normalize`: `none` / `zscore` (特徴次元ごと) / `l2` (ベクトル単位)。既定 `zscore`。
- `position_weight` λ ≥ 0: 正規化パッチ中心 `(cx/W, cy/H)` を λ 倍して特徴に連結。
  λ>0 で「近い位置同士が対応しやすい」空間的事前分布を入れられる。既定 0。

出力: `X ∈ R^{N×d}`, `Y ∈ R^{M×d}` (float32, contiguous, CUDA)。
PCA の基底は `X` と `Y` を連結して求め、両者に同じ射影を適用する。

実装上の取り決め (Phase 2):

- 処理順は 基本特徴 → `normalize` → 位置チャンネル連結。位置チャンネルは正規化の影響を受けない。
- `zscore` の平均・標準偏差も PCA と同様に `X` と `Y` を結合して 1 組だけ求める
  (画像ごとに正規化すると、2画像間の色分布の差が消えるため)。分散 0 の次元は 0 になる。
- `zscore` は標準化のあと **`1/√d` 倍**する (Phase 3 で追加)。二乗距離の平均が次元数に依らず約 2
  (単位ベクトル同士と同じ) になり、OT の `blur` を特徴の種類・次元から独立に選べる。
  無い場合、既定 (`pca` 64 次元) の二乗距離は平均約 128 で `blur=0.05` (ε=0.0025) の 5×10⁴ 倍になり、
  行質量が 10³¹ に発散した。`l2` は単位ベクトルなので二乗距離は 4 以下 (中心化した特徴なら約 2)、`none` は画素スケールのままなので `blur` を手動で選ぶ。
- `pca_lowrank` は乱数を使うため、シードを固定して決定的にする (グローバル RNG は変更しない)。
  `pca_dim` が `min(N+M, 3·patch_size²)` を超える場合は切り詰め、`warnings` に載せる。
- パッチ中心はピクセル端基準: `(col·stride + patch_size/2, row·stride + patch_size/2)`。
  `i = row·cols + col` は `unfold` の出力順 (行優先) と一致する。

### 3.4 OT の解

重み `a_i = 1/N`, `b_j = 1/M` (一様)。コスト `C_ij = cost_scale * ||x_i - y_j||²`
(`half_cost=True` なら `cost_scale=0.5`、既定は 1.0)。

ポテンシャル `(f, g)` は `flash_sinkhorn.sinkhorn_solvers.sinkhorn_flashstyle_symmetric` を
直接呼んで取得する (`SamplesLoss(potentials=True)` の内部実装と同じ関数)。

```python
eps_list = epsilon_schedule(diameter, blur, scaling) + [blur ** 2] * max_final_iters
f, g, n_iters = sinkhorn_flashstyle_symmetric(
    X, Y, a, b,                      # 引数順は (x, y, a, b)
    eps_list=eps_list, cost_scale=cost_scale,   # half_cost=True なら 0.5
    reach_x=reach_x, reach_y=reach_y,
    threshold=threshold * blur ** 2, check_every=inner_iterations,
    allow_tf32=False, autotune=False, return_n_iters=True,
)
```

`SamplesLoss(potentials=True)` を使わない理由 (flash-sinkhorn 0.3.3.post1 で実測):

- potentials 経路は ε スケジュール長 (約 10〜20 反復) で打ち切られ、再構成した計画の行質量が
  `a_i` の 0.8〜5 倍になる。反復数を延ばす `eps_list` 引数はこの経路では無視される。
  `threshold` は絶対値で、ε が小さいと収束前に止まる。
- 上記の直接呼び出しでは最終 ε での追加反復 (`max_final_iters`, 既定 500) を足せる。
  `threshold` は **ε 相対** (`inner_iterations` 反復ごとのポテンシャル最大変化量が
  `threshold * ε` 未満で打ち切り。`None` で無効)。既定 (`blur=0.05`, N=768) では上限まで回り、
  行質量の誤差は約 3〜4% (blur=0.2 では約 0.2〜0.4%)。収束が遅い問題では `converged=False` になる。
- `allow_tf32=False`: 有効だと小さい ε で行質量誤差が約 10% に頭打ちになる。
- `autotune=False`: 有効だと `N`, `M` を 32 で割った値ごとに再チューニングが走り、
  1 形状あたり数秒〜数十秒かかる。無効でも実行時間は同等 (N=3072 で約 0.5 秒)。
- 内部関数への依存なので、`FlashBackend` は必要な引数の有無を起動時に検査し、
  無ければ `BackendUnavailableError` にする。

**注意点 (実装時の必須事項)**

- flash-sinkhorn は CUDA テンソル必須 (CPU を渡すと `BackendUnavailableError`)。
- ポテンシャルは非 debias のもの。デバイアスされたポテンシャルは輸送計画を定義しない。
- ε スケジューリングの **最終 ε は `blur²`** (`half_cost` でも変わらない)。
  計画の再構成にはこの最終値を使う: `eps = blur ** 2`。
- 計画のコストは solver と同じ規約 (`cost_scale * ||x-y||²`) で再構成する。
  食い違うと行質量が大きく崩れる (`tests/test_dense_backend.py` で検証)。
- `potentials` には autograd は付かない (本アプリでは勾配不要)。
- 表示用の Sinkhorn ダイバージェンス値は `SamplesLoss(debias=True, potentials=False)` を
  **別途もう一度** 呼んで取得する。ε スケジュール長の反復だけの近似値で、dense (収束まで反復した
  双対値) との差は約 1〜2%。unbalanced でも取得できる。
- 起動時 (lifespan) に `FlashBackend.warmup()` で Triton カーネルを事前コンパイルする
  (キャッシュ無しで約 8 秒、有りで 1 秒未満。設定 `warmup_on_startup` で無効化でき、失敗しても起動は続ける)。既定 (d=64) 以外の特徴次元や異形サイズは初回のみ約 2 秒かかる。

### 3.5 輸送計画の要約

輸送計画は
`P_ij = a_i * b_j * exp((f_i + g_j - C_ij) / eps)`
で与えられるが、`N×M` を一度に作らず **行チャンク** で処理する
(チャンク幅は `chunk * M * 4 bytes ≤ 64MB` を満たすよう自動決定)。

各チャンクで以下を計算する:

1. `logP_IJ = (f_I[:,None] + g[None,:] - C_IJ) / eps + log a_I[:,None] + log b[None,:]`
2. 行質量 `row_mass_i = exp(logsumexp_j logP_ij)` (balanced なら ≈ `a_i`)
3. 条件付き分布 `p̃_ij = P_ij / row_mass_i`
4. 上位 `top_k` (既定 3) の `(j, p̃_ij, C_ij)`
5. 行エントロピー `H_i = -Σ_j p̃_ij log p̃_ij` (打ち切り前の全列で計算)
6. 信頼度 `confidence_i = max_j p̃_ij`
7. 重心変位 `disp_i = Σ_j p̃_ij * center_B(j) - center_A(i)` (ピクセル単位)

列方向の統計 (`col_mass_j`) も同時に累積し、B 側の「受け取り量」ヒートマップに使う。

**代替経路 (Phase 9)**: 変位場だけなら
`apply_plan_mat_flashstyle` に B 側パッチ中心座標行列を渡すことで
O(nd) メモリのまま重心射影を得られる。チャンク実装のリファレンスとして検証に使う
(`app/ot/projection.py`、`tests/test_projection.py`。API では使わない)。

- カーネルは `mat` に特徴次元と同じ列数 `[M, d]` を要求する。列を `[center_B(j), 1, 0, …]` にして
  `Σ_j P_ij center_B(j)` と行質量を 1 回で得る。`d < 3` なら `x`, `y` をゼロ埋めする。
  ポテンシャルは shifted 形式 `f̂ = f - cost_scale·||x||²`、`ĝ = g - cost_scale·||y||²` で渡す。
- 実測 (RTX 3060、`scripts/bench_ot`): チャンク要約との変位の差は最大 0.02 px、行質量の相対差は 1e-3 未満。
  所要時間は N=M=768 で 0.8 ms (チャンク要約全体は 4.9 ms)、3072 で 7 ms (同 72 ms)。
  top-k・エントロピー・`col_mass` は行全体が要るので、変位だけを置き換えても全体はほぼ速くならない。

### 3.6 ハードマッチングモード (Phase 9, オプション)

`c_transform_fwd(X, Y, psi=g, cost_scale=...)` の argmin `j*(i) = argmin_j [C_ij - g_j]` により、
エントロピー平滑化なしの 1 対 1 寄りの対応を得るモード。UI 上でソフト/ハードを切替 (再実行なし)。

- 計画の行は `P_ij ∝ b_j exp((g_j - C_ij)/ε)` なので、`b` が一様なら `j*` はソフト計画の top-1 と数式上一致する
  (実測 100%、`N≠M` では同値の差で 99.9%)。対応先は変わらず、変わるのは表示量:
  変位は分布の重心ではなく `center_B(j*) - center_A(i)`、受け取り量は `(Σ_{i: j*(i)=j} a_i) / b_j` (選ばれた回数 × M/N)。
- 常に計算して応答に含める (flash は `c_transform_fwd`、dense は行チャンクの argmin。N=M=425 で約 1 ms)。
  `SinkhornBackend.c_transform` として両バックエンドに実装し、`stats.hard_agreement` に top-1 との一致率を載せる。

### 3.7 CPU リファレンス実装 (`DenseTorchBackend`)

`N×M` コスト行列を実体化する log-domain Sinkhorn。flash と同じ ε 列・同じ対称更新で、
ポテンシャルは約 4×10⁻⁶ で一致する (unbalanced も同様)。ダイバージェンスは balanced のみ (`None`)。
用途:

- CUDA が無い環境での動作継続 (`MAX_PATCHES_CPU` = 1024 に制限)
- CI テストでの数値検証 (GPU 不要)
- flash-sinkhorn 出力との一致検証 (ポテンシャル差 / top-1 一致率)

## 4. API 仕様

すべて `/api` 配下。エラーは `{"detail": {"code": str, "message": str, "hint": str|null}}`。

### 4.1 `GET /api/health`

```json
{
  "status": "ok",
  "device": "cuda:0",
  "cuda_available": true,
  "gpu_name": "NVIDIA GeForce RTX 3060",
  "torch_version": "2.x.x",
  "flash_sinkhorn_version": "0.3.3.post1",
  "default_backend": "flash",
  "feature_types": ["pca", "raw", "color", "dinov2"],
  "limits": {"max_patches": 4096, "max_upload_bytes": 10485760}
}
```

### 4.2 `POST /api/images` (multipart/form-data, field `file`)

制約: PNG/JPEG/WebP、10MB 以下、辺 ≤ 8192px。
`width`/`height` は前処理後、`original_*` は EXIF 回転後・縮小前の寸法。

```json
{"image_id": "01J...", "width": 512, "height": 384,
 "original_width": 1920, "original_height": 1440, "url": "/api/images/01J..."}
```

### 4.3 `GET /api/images/{image_id}`

前処理後の PNG を返す (フロントはこれを表示し、座標系をバックエンドと一致させる)。

### 4.4 `POST /api/match`

リクエスト:

```json
{
  "image_a": "01J...",
  "image_b": "01J...",
  "patch": {"size": 16, "stride": 16},
  "feature": {"type": "pca", "pca_dim": 64, "normalize": "zscore", "position_weight": 0.0},
  "ot": {
    "blur": 0.05, "scaling": 0.5, "half_cost": false,
    "reach_x": null, "reach_y": null,
    "threshold": 1e-3, "inner_iterations": 10,
    "backend": "auto"
  },
  "output": {"top_k": 3, "min_weight": 1e-4, "compute_divergence": true}
}
```

レスポンス:

```json
{
  "grid_a": {"rows": 24, "cols": 32, "patch_size": 16, "stride": 16,
             "image_width": 512, "image_height": 384},
  "grid_b": {"...": "同上"},
  "matches": [
    {"i": 0, "confidence": 0.71, "entropy": 1.34, "row_mass": 1.0,
     "displacement": [12.0, -4.0],
     "targets": [{"j": 33, "weight": 0.71, "cost": 0.031},
                 {"j": 34, "weight": 0.18, "cost": 0.052}],
     "hard": {"j": 33, "cost": 0.031, "displacement": [16.0, 0.0]}}
  ],
  "col_mass": [1.02, 0.97, "..."],
  "hard_col_mass": [1.0, 2.0, 0.0, "..."],
  "stats": {
    "n_patches_a": 768, "n_patches_b": 768, "feature_dim": 64,
    "eps": 0.0025, "sinkhorn_divergence": 0.0123,
    "backend": "flash", "device": "cuda:0",
    "hard_agreement": 1.0,
    "elapsed_ms": {"preprocess": 41, "features": 18, "solve": 96, "plan": 35, "hard": 1, "total": 190}
  },
  "warnings": []
}
```

リクエストの取り決め (Phase 4):

- 未知のキーは拒否する (`INVALID_PARAMS`)。`patch` / `feature` / `ot` / `output` は省略でき、省略時は上記の既定値。
- `patch.stride` を省略すると `patch.size`。`ot.backend` を省略するとサーバ設定 `default_backend`。
- 範囲: `patch.size` 2..64、`feature.pca_dim` 1..256、`ot.blur` (0, 10]、`ot.scaling` (0, 1)、
  `ot.reach_*` > 0、`ot.threshold` > 0 または `null`、`ot.inner_iterations` 1..1000、
  `output.top_k` 1..10、`output.min_weight` [0, 1)。
- 最終 ε での追加反復の上限 `max_final_iters` は API に出さず、サーバ設定 (既定 500)。
- `ot.backend="flash"` を明示して CUDA / flash-sinkhorn が使えない場合は `INVALID_PARAMS` (422)。`auto` は dense へフォールバックする。
- A・B のパッチ数の上限は、CUDA があれば `MAX_PATCHES`、無ければ `MAX_PATCHES_CPU` (どちらの `backend` でも同じ)。

レスポンスの取り決め (Phase 4):

- `matches` は A 側パッチのインデックス `i` 昇順。`i = row * cols + col`。
- `targets` は `weight` 降順、`min_weight` 未満は除外 (0 件もありうる)。
- `col_mass` は B 側の受け取り質量を `b_j` で割って正規化した値 (1.0 が期待値)。
  `row_mass` も同様に `a_i` で割った値 (balanced なら ≈ 1.0)。
- `stats` は上記に加えて `iterations`, `converged` (`threshold` で打ち切られたか)、
  `row_mass_error` (balanced のときの `max_i |row_mass_i - 1|`、unbalanced は `null`) を持つ。
  `sinkhorn_divergence` は `compute_divergence=false` またはバックエンド未対応 (dense の unbalanced) なら `null`。
- `stats.elapsed_ms` のキーは `preprocess` (パッチ抽出。画像のデコードと GPU セマフォの待ち時間は含まない)、
  `features`, `solve`, `divergence` (無効なら 0), `plan`, `hard`, `total`。GPU は各段階の前後で同期して計測する。
- `hard` / `hard_col_mass` / `stats.hard_agreement` はハード割当 (3.6、Phase 9)。`hard_col_mass` は `col_mass` と同じ規約。
- `warnings` は `{"code", "message", "params"}` の配列 (Phase 8)。`message` は英語の説明で、フロントは `code` と `params` から
  日本語の文言を作る (未知のコードは `message` をそのまま出す)。コードは次の 3 つ:
  - `PCA_DIM_REDUCED` (`requested`, `actual`): PCA 次元の切り詰め。
  - `ROW_MASS_ERROR` (`error`, `tolerance`, `converged`): balanced で `row_mass_error` が設定 `warn_row_mass_error` (既定 0.05) を超えた。
  - `NOT_CONVERGED` (`iterations`): unbalanced で `converged=false`。
  行質量が崩れても `p̃` は行ごとに正規化されるので表示は成立する。

### 4.4.1 同梱サンプル (Phase 8)

- `GET /api/samples`: `[{"id", "title", "description"}]`。`assets/samples/samples.json` (設定 `samples_dir`) の一覧。無ければ空。
- `POST /api/samples/{id}`: サンプルの 2 枚をアップロードと同じ前処理で画像ストアに登録し、
  `{"a": <ImageUploadResponse>, "b": <ImageUploadResponse>}` を返す。未知の `id` は 404 `HTTP_ERROR`。
- サンプル画像は scikit-image 同梱の CC0 / パブリックドメインの写真から `backend/scripts/make_samples.py` で作る (出所は `assets/samples/CREDITS.md`)。

### 4.5 エラーコード

| code | HTTP | 条件 |
|---|---|---|
| `IMAGE_TOO_LARGE` | 413 | アップロードサイズ超過 |
| `UNSUPPORTED_FORMAT` | 415 | デコード不可 |
| `IMAGE_NOT_FOUND` | 404 | `image_id` 不明 / TTL 切れ |
| `TOO_MANY_PATCHES` | 422 | `N` または `M` が上限超過 (`hint` に、両画像が収まる最小の `patch.size` を示す) |
| `INVALID_PARAMS` | 422 | パラメータ範囲外 (Pydantic)、パッチが画像より大きい、明示したバックエンドが使えない、`dinov2` で extra `deep` が未導入 |
| `SOLVER_FAILED` | 500 | CUDA OOM / Triton コンパイル失敗 / ポテンシャルや計画が非有限値に発散 |
| `HTTP_ERROR` | 404/405 等 | 未定義ルート・メソッド違反 (仕様外の汎用コード) |
| `INTERNAL_ERROR` | 500 | 想定外の例外 (仕様外の汎用コード) |
| `INTERNAL_ERROR` | 503 | `dinov2` のモデル (重み) を読み込めない (オフラインなど) |

補足: 辺が `max_image_side` (8192px) 超の画像は `IMAGE_TOO_LARGE`、
前処理後に辺が `stride` 未満になる画像は `INVALID_PARAMS`、
`image_id` の形式不正も `IMAGE_NOT_FOUND` として返す (ID の妥当性を漏らさない)。

### 4.6 画像ストア

- 一時ディレクトリに `image_id` 単位で保存。TTL 既定 1時間、起動時とリクエスト時に掃除。
- `image_id` は ULID。ディレクトリトラバーサル防止のため厳密に検証する。

## 5. フロントエンド仕様

### 5.1 技術構成

React 19 + TypeScript + Vite、状態管理は React 標準 hooks + TanStack Query、
スタイルは Tailwind CSS。描画は Canvas 2D (画像・ヒートマップ) と SVG (線・枠) の併用。
重量級の描画ライブラリは使わない。

### 5.2 画面構成 (単一ページ)

```
┌──────────────────────────────────────────────────────────┐
│ ヘッダー: タイトル / GPU バッジ (health) / 実行ボタン      │
├───────────────────────────────┬──────────────────────────┤
│ 画像A (drop zone → canvas)     │ 画像B (drop zone → canvas)│
│  ・パッチ格子オーバーレイ       │  ・対応パッチのハイライト  │
│  ・hover したパッチを強調       │  ・重みに応じた不透明度    │
├───────────────────────────────┴──────────────────────────┤
│ 接続線オーバーレイ (2画像をまたぐ SVG レイヤ)             │
├───────────────────────────────┬──────────────────────────┤
│ パラメータパネル                │ 統計パネル               │
│  patch/feature/ot/output       │  divergence, 時間内訳     │
└───────────────────────────────┴──────────────────────────┘
```

### 5.3 可視化モード

| モード | 内容 |
|---|---|
| `hover` (既定) | A のパッチにホバー → B 側で top-k を重み順の不透明度で表示 + 接続線 |
| `confidence` | A 側を `confidence` のヒートマップで着色 (低 = 曖昧な対応) |
| `entropy` | A 側を行エントロピーで着色 |
| `flow` | A 側各パッチから `displacement` 方向の矢印を描画 (間引き表示可) |
| `coverage` | A 側を `row_mass`、B 側を `col_mass` で着色 (unbalanced の効果が見える) |
| `pin` | クリックでパッチを固定し複数同時比較 |

実装上の取り決め (Phase 7):

- ヒートマップ系 (`confidence` / `entropy` / `coverage`) でも hover の枠と接続線は併用する。クリック / Enter での pin は
  `pin` とヒートマップ系のみ。`hover` では受け付けず、`flow` では矢印と重なるので pin も hover の接続線も出さない (hover は A 側の枠のみ)。
  固定済みの pin は状態として残り、pin を受け付けるモードに戻ると再び描かれる。
- 色は viridis。範囲は値の両端 2% を飽和させた分位点で、凡例に実際の範囲を出す。`coverage` は balanced だと 1.0 の近傍の数値誤差が
  強い模様になるので、1.0 の上下 ±0.1 を最低幅とする。
- `coverage` の A 側 (送出量 `row_mass`) は Phase 9 で追加。色範囲は A・B の値をまとめて決める (同じ色が同じ値)。
  ハード割当では A の各パッチがちょうど 1 回送るので B 側だけ塗る。
- unbalanced の結果 (`row_mass_error` が `null`) では、`flow` の矢印と pin の対応先を送出量で薄くする
  (不透明度の係数 0.12 + 0.88·√min(row_mass, 1))。条件付き確率は行ごとに正規化した量なので、送出量が 0 に近い
  パッチでも濃い対応に見えてしまうため。`flow` は確信度で薄くする値との小さい方を取る。hover は薄くせず、送出量の数値を出す。
  balanced でも未収束だと `row_mass` は 1.0 から数十 % ずれる (実測: 左右反転サンプルで 0.71〜1.23) ので、balanced では薄くしない。
- `flow` の矢印は A 側パッチ中心から `displacement` (画像ピクセル) × 表示倍率。A の画像領域で切り取り、色は変位の大きさ。
  間引きは行・列とも k パッチおき (1〜8、既定 2)。色範囲は間引きに依存しない。
  「確信度で薄く」(既定 on、Phase 8) では、確信度を色範囲と同じ分位点で [0, 1] に写し、不透明度 0.12〜1 にする
  (対応先の無い端のパッチが、拡散した分布の重心を指す長い矢印になって全体を読みにくくするため)。
- 接続線の濃さ・太さ・色の基準は「相対 (各パッチの top-1 を 1、既定)」と「絶対 (条件付き確率そのまま)」を切り替えられる。
- パラメータは `patch` / `feature` / `ot` / `output` の全項目を明示して送る (入力欄は文字列で保持し、範囲はバックエンドのスキーマに合わせて
  クライアントでも検証する。不正値があると「実行」は無効)。`ot.backend` の「サーバ設定」は `null` で送る。
- 統計パネルは、同じ画像ペアで直前に成功した実行との数値差と、表示中の結果が現在のパラメータと異なる旨を出す。警告はコード別の日本語で表示する (Phase 8)。
- 結果 JSON は `{"request": ..., "response": ...}`。可視化 PNG は A・B を左右に並べ (画像 1 ピクセル = 2 出力ピクセル)、格子・ヒートマップ・
  矢印・pin の接続線を重ねる。hover は一時的なので含めない。凡例は幅が足りなければ 2 段にし、それでも入らなければ文字を縮める。

### 5.4 その他

- パラメータ変更は即実行せず「実行」ボタンで明示的に送信。実行中はスピナーと中断ボタン。
- 結果 JSON のダウンロード、可視化 PNG の書き出し。
- 画像未選択・エラー時のメッセージ表示。Phase 8 で次を追加:
  - 実行前に分かる失敗 (画像未選択・不正なパラメータ・CUDA が無いのに `flash`・パッチ数超過) は「実行」を無効にして理由を出す
    (パッチ数超過は、両画像が収まる最小の patch size を添える)。
  - サーバ停止中は本文先頭に起動コマンドと「再接続」を出す。`/api/match` が `NETWORK_ERROR` になったら health を即再取得する。
  - GPU が無いときは、CPU (dense) で計算することとパッチ数の上限を案内する。
  - 同梱サンプルをワンクリックで A・B に読み込むボタン。
- レスポンシブ: 幅が狭い場合は上下2段レイアウトに切替。
- 開発時は Vite の `server.proxy` で `/api` を `http://localhost:8000` へ転送。

## 6. ディレクトリ構成

```
flash-sinkhorn-webui/
├── SPEC.md
├── TODO.md
├── README.md
├── backend/
│   ├── pyproject.toml            # uv 管理
│   ├── app/
│   │   ├── main.py               # FastAPI アプリ生成・CORS・例外ハンドラ
│   │   ├── config.py             # pydantic-settings
│   │   ├── api/routes/{health,images,match,samples}.py
│   │   ├── schemas/{common,health,image,match,sample}.py
│   │   ├── services/{image_store,preprocess,patches,features,matching,notices,samples,device}.py
│   │   └── ot/{base,flash_backend,dense_backend,plan,schedule}.py
│   ├── scripts/{gpu_smoke,bench_ot,bench_api,make_samples}.py
│   └── tests/
│       ├── test_patches.py  test_features.py
│       ├── test_plan.py     test_backend_parity.py   # GPU 有無で skip
│       └── test_api.py      test_samples.py
├── assets/samples/               # 同梱サンプル (samples.json + 画像 + CREDITS.md)
├── docs/screenshot.png
└── frontend/
    ├── package.json  vite.config.ts  playwright.config.ts  .prettierrc.json
    ├── e2e/{match,errors}.spec.ts  # Playwright
    └── src/
        ├── main.tsx  App.tsx
        ├── api/{client.ts,types.ts}
        ├── components/*.tsx       # ImageDropzone, ImagePanel, MatchStage, MatchOverlay, ParamPanel, StatsPanel, ModeSelector, ...
        ├── hooks/{useHealth,useImageUpload,useMatch,useOverlayGeometry}.ts
        └── lib/*.ts               # grid, colormap, overlay, heatmap, flow, draw, export, legend, params, preflight, warnings, ...
```

## 7. 依存パッケージ (すべて公開パッケージ)

**backend** (`uv` / `pyproject.toml`)

- ランタイム: `fastapi`, `uvicorn[standard]`, `python-multipart`, `pydantic`,
  `pydantic-settings`, `pillow`, `numpy`, `torch>=2.5`, `triton>=3.1`,
  `flash-sinkhorn>=0.3.3`
- 開発: `pytest`, `pytest-asyncio`, `httpx`, `ruff`, `mypy`

torch は CUDA ビルドを公開インデックス (`https://download.pytorch.org/whl/cu12x`)
から取得する。`~/workspace/flash-sinkhorn` のローカルソースは **参照のみ**で、
依存には使わず PyPI の `flash-sinkhorn` を入れる。

**frontend** (`npm`)

- `react`, `react-dom`, `@tanstack/react-query`
- dev: `vite`, `@vitejs/plugin-react`, `typescript`, `tailwindcss`,
  `@types/react`, `@types/react-dom`, `oxlint`, `prettier`, `vitest`, `@playwright/test`
  (lint は eslint ではなく oxlint。Phase 5 で採用し、Phase 8 で確定)

## 8. 非機能要件

| 項目 | 目標 |
|---|---|
| レイテンシ | 512px / patch 16 (N=M=768) で end-to-end 1秒以内 (GPU) |
| メモリ | 計画の再構成は 64MB/チャンク以内。GPU メモリは 2GB 以内 |
| 同時実行 | GPU 計算は `asyncio` のスレッドプールで実行し、セマフォで同時 1件に制限 |
| ログ | リクエストごとに `image_id`, パラメータ, 各段階の所要時間を構造化ログ出力 |
| セキュリティ | アップロードは Pillow で再デコードして保存、拡張子ではなく中身で判定 |

## 9. テスト方針

- **単体**: パッチ抽出のインデックス対応 (`i ↔ (row,col) ↔ 画素座標`)、
  特徴正規化、計画チャンクの結果が非チャンク計算と一致すること。
- **数値**: 同一画像を2枚渡すと対応がほぼ対角 (top-1 一致率 ≥ 95%) になること。
  既知の平行移動画像で `displacement` が真値に近いこと (中央値誤差 ≤ 1パッチ)。
- **バックエンド整合**: 小サイズ (N=M=256) で `flash` と `dense` の
  ポテンシャル差と top-1 一致率を比較 (GPU がある環境のみ実行、無ければ skip)。
- **API**: `TestClient` でアップロード→マッチの一連を検証 (`dense` バックエンド強制)。
- **フロント**: `vitest` で grid 座標変換と colormap などの純関数を単体テスト。E2E は Playwright (`npm run e2e`。
  アップロード → 実行 → ホバーでのハイライトと、API をモックしたエラー系 UI)。

## 10. 実装順序と段階的マイルストン

| マイルストン | 内容 | 完了条件 |
|---|---|---|
| M1 | バックエンド骨格 + 画像アップロード | `/api/health` と画像往復が動く |
| M2 | マッチングコア (CPU dense) | CLI/pytest で対応が取れる |
| M3 | flash-sinkhorn 統合 | GPU で M2 と整合する結果、速度目標達成 |
| M4 | `/api/match` 公開 | curl で完全なレスポンスが得られる |
| M5 | フロント骨格 + 画像表示 | 2枚アップロードして並べて表示 |
| M6 | 可視化 (hover / 接続線) | パッチ対応が目視できる |
| M7 | ヒートマップ・フロー・パラメータUI | 全モード動作 |
| M8 | 仕上げ (エラー処理・README・E2E) | 初見ユーザが手順書どおり動かせる |

詳細な作業項目は `TODO.md` を参照。
