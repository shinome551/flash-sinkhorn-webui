# TODO: flash-sinkhorn-webui

`SPEC.md` の実装タスク一覧。各フェーズは独立してレビュー可能な単位になっている。
`[ ]` が未着手、`[x]` が完了。各フェーズの **完了条件** を満たしたら次へ進む。

---

## Phase 0: 環境セットアップ

- [x] `git init` とベースの `.gitignore` (Python/Node/画像一時ファイル)
- [x] `backend/pyproject.toml` を作成し、`uv sync` で仮想環境を作る
      (依存は SPEC 7章。torch は CUDA ビルドを公開インデックスから)
- [x] GPU スモークテスト: `SamplesLoss(potentials=True)` を 1024×1024 の
      ランダム点群で実行し、`f, g` の形状と所要時間を記録する
- [x] `frontend/` を `npm create vite@latest -- --template react-ts` で生成し、
      Tailwind と TanStack Query を導入、dev サーバ起動を確認
- [x] `README.md` に起動手順の骨子を書く (以降のフェーズで加筆)

**完了条件**: バックエンドとフロントエンドの両方が起動し、GPU で flash-sinkhorn が動く。

---

## Phase 1: バックエンド骨格と画像入出力 (M1)

- [x] `app/config.py`: `pydantic-settings` で設定 (一時ディレクトリ、TTL、
      `max_upload_bytes`, `max_patches`, `resize_max`, `default_backend`)
- [x] `app/main.py`: FastAPI 生成、CORS (dev: localhost:5173)、
      共通例外ハンドラ (SPEC 4.5 の `{"detail": {code, message, hint}}` 形式)
- [x] `app/schemas/common.py`: エラーレスポンス、`ErrorCode` 列挙
- [x] `app/api/routes/health.py`: `GET /api/health` (SPEC 4.1 のフィールド一式)
- [x] `app/services/image_store.py`: ULID 採番、保存、取得、TTL 掃除、
      `image_id` の厳密検証 (パストラバーサル防止)
- [x] `app/services/preprocess.py`: EXIF 回転、RGB 変換、アルファ合成、
      `resize_max` での縮小、`stride` 倍数への切り詰め
- [x] `app/api/routes/images.py`: `POST /api/images`, `GET /api/images/{id}`
- [x] `tests/test_api.py`: アップロード→取得の往復、サイズ超過 413、非画像 415

**完了条件**: `curl -F file=@a.png localhost:8000/api/images` が ID を返し、
返った URL で前処理後の画像が取得できる。

---

## Phase 2: パッチ抽出と特徴ベクトル化 (M2 前半)

- [x] `app/services/patches.py`
  - [x] `extract_patches(img, size, stride) -> (patches[N, C*P*P], grid_meta)`
  - [x] `patch_centers(grid_meta) -> [N, 2]` (ピクセル座標)
  - [x] `i ↔ (row, col)` 変換ユーティリティ
- [x] `app/services/features.py`
  - [x] `raw` / `pca` / `color` の3種 (`pca` は X,Y 結合で基底を作り共有)
  - [x] `normalize`: `none` / `zscore` / `l2`
  - [x] `position_weight` による位置チャンネル連結
  - [x] 出力は float32 contiguous、指定デバイスへ転送
- [x] `tests/test_patches.py`: パッチ数、中心座標、既知パターン画像での画素一致
- [x] `tests/test_features.py`: 次元、正規化後の統計、PCA 基底の共有を確認

**完了条件**: 画像 → `X[N,d]` が決定的に得られ、インデックス対応がテストで保証される。

---

## Phase 3: OT コア (M2 後半 + M3)

- [x] `app/ot/base.py`: `SinkhornBackend` Protocol
      (`solve_potentials(X, Y, a, b, params) -> (f, g, eps, info)` と `divergence`)
- [x] `app/ot/dense_backend.py`: log-domain Sinkhorn (ε スケジュール同一、CPU/GPU)
- [x] `app/ot/flash_backend.py`: flash-sinkhorn ラッパ
  - [x] `sinkhorn_flashstyle_symmetric` を直接呼んで `(f, g)` を取得
        (`SamplesLoss(potentials=True)` は反復不足で行質量が崩れるため。SPEC 3.4 参照)
  - [x] `eps = blur ** 2` を返す (ε スケジューリングの最終値)
  - [x] CUDA 未使用時は明示的に例外 (`BackendUnavailableError`)、`get_backend("auto")` は dense へフォールバック
  - [x] `debias=True` の別呼び出し (`SamplesLoss`) で Sinkhorn ダイバージェンスを取得
  - [x] 起動時ウォームアップ `FlashBackend.warmup()` (呼び出しの配線は Phase 4)
- [x] `app/ot/plan.py`: 行チャンクでの計画要約
  - [x] チャンク幅の自動決定 (`chunk * M * 4 ≤ 64MB`)
  - [x] `row_mass`, `top_k`, 行エントロピー, `confidence`, 重心変位, `col_mass`
  - [x] `min_weight` による打ち切り (`top_valid` マスクで返す)
- [x] 特徴のスケール調整: `zscore` 後に `1/√d` 倍 (注意点8の解消。SPEC 3.3)
- [x] `tests/test_plan.py`: チャンクあり/なしの一致、行質量 ≈ `a_i`、
      同一画像での top-1 対角一致率 ≥ 95%
- [x] `tests/test_backend_parity.py`: `flash` vs `dense` (GPU 無しなら skip)
- [x] ベンチ: N=M=768 / 3072 での `solve` と `plan` の所要時間を記録 (`uv run python -m scripts.bench_ot`)

**ベンチ結果** (RTX 3060, torch 2.11+cu128, flash-sinkhorn 0.3.3.post1, 特徴 = pca 64 + zscore, 合成テクスチャ):

| N=M | blur | backend | solve | plan | 行質量の最大誤差 |
|---|---|---|---|---|---|
| 768 | 0.05 | flash | 126 ms (512 反復) | 5 ms | 3.4% |
| 768 | 0.2 | flash | 131 ms | 4 ms | 0.2% |
| 3072 | 0.05 | flash | 477 ms | 70 ms | 41% |
| 3072 | 0.2 | flash | 471 ms | 69 ms | 1.0% |
| 768 | 0.05 | dense (GPU) | 404 ms | 4 ms | 3.4% |
| 3072 | 0.05 | dense (GPU) | 1545 ms | 73 ms | 41% |

- SPEC 8章の目標 (N=M=768 で 1 秒以内) は solve+plan で約 130 ms。ダイバージェンス込みでも +17 ms。
- 収束が遅い問題では反復上限 (`max_final_iters=500`) まで回り `converged=False` になる。
  N=3072・blur=0.05 の最大誤差 41% は上限を 1000/2000 に上げても 13% で頭打ち (時間は 2/4 倍)。
  条件付き分布 `p̃` は行ごとに正規化されるので表示は成立するが、Phase 4 で警告に載せる。

**完了条件**: pytest が GPU 環境で全緑、CPU 環境でも GPU 依存テストの skip 以外は全緑。
(確認済み: GPU 環境 143 passed / `CUDA_VISIBLE_DEVICES=""` で 129 passed, 14 skipped)

**Phase 4 への持ち越し**

- `BackendUnavailableError` を API エラーに変換する (`backend="flash"` 明示で CUDA 無し → 422 `INVALID_PARAMS` 相当)。
- `SolveInfo.converged == False` や行質量の誤差が大きいとき `warnings` に載せる。
- `max_final_iters` は API に出さず `config.py` の設定にする (既定 500)。
- `create_app` の lifespan から、flash が使える場合に `FlashBackend.warmup()` を呼ぶ。
- `compute_divergence` 用に `backend.divergence()` を呼ぶ (dense の unbalanced は `None`)。

---

## Phase 4: マッチング API の公開 (M4)

- [x] `app/schemas/match.py`: `MatchRequest` / `MatchResponse` (SPEC 4.4 準拠、
      範囲バリデーション: `blur>0`, `0<scaling<1`, `top_k 1..10` 等)
- [x] `app/services/matching.py`: 前処理→パッチ→特徴→OT→要約 のオーケストレーション。
      各段階の所要時間を計測して `stats.elapsed_ms` に格納
- [x] `app/api/routes/match.py`: `POST /api/match`。
      GPU 計算は `run_in_threadpool` + セマフォ(同時1件)で実行
- [x] `TOO_MANY_PATCHES` / `SOLVER_FAILED` (CUDA OOM 含む) のハンドリング
- [x] 構造化ログ (image_id, パラメータ, 各段階時間)
- [x] `tests/test_api.py` 拡充: `backend="dense"` 指定での end-to-end
- [x] Phase 3 からの持ち越し (すべて対応)
  - [x] `BackendUnavailableError` → 422 `INVALID_PARAMS`
  - [x] 行質量の誤差 (既定 5% 超) と、unbalanced での `converged=False` を `warnings` に載せる
  - [x] `max_final_iters` を `config.py` へ (API には出さない)
  - [x] lifespan で `FlashBackend.warmup()` (`warmup_on_startup` で無効化可)
  - [x] `backend.divergence()` を `compute_divergence` で呼ぶ

**完了条件**: curl で 2枚の画像 ID を投げると SPEC どおりの完全な JSON が返る。
(確認済み: 512×384 の合成ペア N=M=768 で `total` 約 130 ms (flash, RTX 3060)、
pytest は GPU 環境 172 passed / `CUDA_VISIBLE_DEVICES=""` で 157 passed, 15 skipped)

**Phase 5 以降への持ち越し**

- 既定パラメータ (`blur=0.05`) でも、テクスチャの乏しい画像では行質量誤差が 5% を超えて `warnings` が出る
  (合成ペアで 9.6%)。フロントは警告を握りつぶさず StatsPanel に出すこと。閾値は `FSW_WARN_ROW_MASS_ERROR`。
- `row_mass` は `a_i` で割った値 (≈1.0)。`col_mass` と同じ規約。
- GPU セマフォは `asyncio.Semaphore(1)`。クライアント切断で待機側が取り消されても、実行中のスレッドは止まらない
  (Phase 6 の「中断」は fetch の abort のみで、サーバ側の計算は完走する)。

---

## Phase 5: フロントエンド骨格 (M5)

- [x] `src/api/types.ts`: バックエンドのスキーマに対応する TypeScript 型
- [x] `src/api/client.ts`: `uploadImage` / `postMatch` / `getHealth` (エラー整形込み)
      (`uploadImage` は進捗を出すため XHR。`ApiError` に `NETWORK_ERROR` / `UNKNOWN` をクライアント側コードとして追加)
- [x] `vite.config.ts` に `/api` の dev プロキシ設定
- [x] `components/ImageDropzone.tsx`: drag&drop + ファイル選択、プレビュー、
      アップロード進捗、エラー表示
- [x] `components/ImagePanel.tsx`: 返却画像を canvas に描画し、
      表示スケール (`displayScale`) を管理 (`children(displayScale)` で上に重ねる要素を受ける。Phase 6 の入口)
- [x] `lib/grid.ts`: `i ↔ (row,col) ↔ 表示座標` の変換 (+ `vitest` で単体テスト)
      (backend の `compute_grid` / `patch_centers` の実出力を期待値に使用。hover 用の `patchAt` も含む)
- [x] ヘッダーに `/api/health` の GPU バッジを表示 (GPU / CPU / サーバ停止中 の 3 状態)
- [x] 全体レイアウト (Tailwind) とレスポンシブ切替 (`lg` 未満は縦積み)

**完了条件**: 2枚アップロードして並べて表示でき、パッチ格子の座標変換が検証済み。
(確認済み: `npm test` 23 passed、`tsc -b` / `oxlint` / `npm run build` 通過。
Playwright で 2 枚のアップロードと表示、非画像の 415 表示、幅 600px の縦積み、バックエンド停止時のバッジを目視確認)

**Phase 6 以降への持ち越し**

- パッチ格子は `App.tsx` で `DEFAULT_PATCH_SIZE = 16` から `computeGrid` して描いている。
  マッチ結果が返ったら `MatchResponse.grid_a/grid_b` を使うこと (Phase 7 の ParamPanel で patch size を可変にする)。
- 保存期限切れ (TTL) で `GET /api/images/{id}` が失敗すると、`ImageSlot` が画像を外して再アップロードを促す。
  `postMatch` が `IMAGE_NOT_FOUND` を返した場合の扱いは Phase 6 で同様にする。
- フロントの lint は `oxlint`、テストは `vitest` (SPEC 7章の eslint/prettier とは異なる。Phase 8 で整理)。

---

## Phase 6: マッチ可視化 (M6)

- [x] `hooks/useMatch.ts`: 実行ボタンでマッチング送信、ローディング/エラー/中断
      (新規実行で前の要求を abort、`reset` で中断+結果消去。ヘッダーの「実行 / 中断」ボタンから使う)
- [x] `components/MatchOverlay.tsx` (+ `PatchPicker.tsx` / `MatchStage.tsx` / `lib/overlay.ts`)
  - [x] A 側のパッチ格子と hover 検出 (`PatchPicker` が A の上の透明な入力層。`patchAtDisplay` を使用)
  - [x] B 側で top-k を重み順の不透明度でハイライト
  - [x] 2画像をまたぐ SVG 接続線 (太さ・不透明度・色を重みに連動。hover は重み→viridis、pin は識別色)
  - [x] クリックでの `pin` (複数固定・個別解除。色は slot 単位で解除しても変わらない。チップと「すべて解除」)
- [x] `lib/colormap.ts`: viridis 相当の連続カラーマップ (+ `vitest`)
- [x] キーボード操作 (A にフォーカスして矢印 = 移動、Shift+矢印 = 4 パッチ、Enter/Space = pin、Esc = 解除)
- [x] Phase 5 からの持ち越し: `IMAGE_NOT_FOUND` は両画像を GET して無い方だけ外し、スロットに案内を出す

**完了条件**: ホバーで対応パッチと接続線が滑らかに描画される (60fps 目標)。
(確認済み: Playwright で hover / pin / 再クリック解除 / 矢印+Enter+Esc / 幅 600px の縦積み / TTL 切れ / 中断 / サーバ停止を目視確認。
画像 A 上で 600 回マウスを動かしてもフレーム間隔は最大 16.8ms。`npm test` 49 passed、`tsc -b` / `oxlint` / `npm run build` 通過)

**Phase 7 以降への持ち越し**

- 選択状態 (`active` / `pins`) は `MatchStage` が持つ。Phase 7 の `ModeSelector` では `hover` / `pin` をここに繋ぎ、
  ヒートマップ系のモードでは `PatchPicker` を無効化するか併用するかを決める。
- `MatchSummary` (時間・N・警告) は `StatsPanel` までのつなぎ。警告は握りつぶさずそこへ移す。
- パッチサイズは `App.tsx` の `DEFAULT_PATCH_SIZE` 固定 (実行時も明示して送る)。ParamPanel で可変にする。
- 既定パラメータだと条件付き確率が拡散しやすく (合成ペアで top-1 の重みが 0.03〜0.08)、接続線は薄く細くなる。
  `weightOpacity` は絶対値ベースなので、見づらければモードや正規化 (top-1 に対する相対値) を Phase 7 で検討する。
- vitest は `environment: node` でコンポーネントは対象外。UI の回帰は Phase 8 の Playwright E2E で担保する。

---

## Phase 7: 可視化モードとパラメータ UI (M7)

- [x] `components/ModeSelector.tsx`: `hover` / `confidence` / `entropy` /
      `flow` / `coverage` / `pin` の切替 (`lib/modes.ts` にモードごとの振る舞いを集約)
  - ヒートマップ系は hover・pin・接続線と併用。`hover` / `flow` では pin を受け付けず、固定済みの pin も描かない (切り替えると復帰)
  - `flow` は hover で A 側の枠だけ (接続線は矢印と重なるので出さない)
- [x] `confidence` / `entropy` ヒートマップ (A 側)、`coverage` (B 側 `col_mass`)
      (`lib/heatmap.ts` + `HeatmapLayer` (canvas) + `ColorBar` 凡例。色範囲は両端 2% を飽和させた分位点、coverage は 1.0 の上下 ±0.1 を最低幅にする)
- [x] `flow` モード: `displacement` の矢印描画 + 間引き密度スライダ (`lib/flow.ts`。矢印は A の画像領域で切り取り、色は変位の大きさ)
- [x] `components/ParamPanel.tsx`: patch / feature / ot / output の各パラメータ。
      既定値・範囲・ツールチップ付きの説明 (blur, reach 等。`lib/params.ts` が検証とリクエスト変換、不正値は「実行」を無効化)
      (reach_x / reach_y も出した。Phase 9 の「reach の UI 露出」は解説の拡充のみ。`debias` は API のパラメータではないので
      `compute_divergence` のツールチップで説明)
- [x] `components/StatsPanel.tsx`: ダイバージェンス、パッチ数、ε、時間内訳、警告 (`MatchSummary` は廃止)
      + 直前の実行との数値差 (同じ画像ペアの間。`useMatch.previousStats`) と、結果が現在の設定と異なるときのバナー
- [x] 結果 JSON (`{request, response}`) のダウンロードと可視化 PNG の書き出し (`lib/export.ts`。hover は含めず、pin は含める)
- [x] Phase 6 からの持ち越し
  - [x] 線の濃さの基準: 「相対 (top-1 = 最大、既定) / 絶対」を切替
  - [x] パッチサイズを ParamPanel で可変に (プレビュー格子と実行時の両方に反映。パッチ数の目安と上限超過の警告つき)

**完了条件**: 全モードが動作し、パラメータ変更 → 再実行 → 差分が目視できる。
(確認済み: Playwright で hover / 確信度 / フロー / カバレッジ / pin、reach 設定での再実行 (coverage で A に対応のない B の端が受け取り 0 として見える)、
不正値のエラー表示と実行ボタンの無効化、JSON / PNG のダウンロード内容、幅 600px の縦積みを目視確認。
`npm test` 86 passed、`tsc -b` / `oxlint` / `npm run build` 通過)

**Phase 8 以降への持ち越し**

- flow: 対応のない端のパッチは拡散した分布の重心を指すので、変位が数百 px の長い矢印になり全体が読みにくい (合成ペアで確認)。
  確信度で矢印を薄くする案があるが未実装 (仕様外なので要判断)。
- PNG の凡例は、画像が非常に小さい (幅 128px 程度) とラベルと値が重なる。
- ParamPanel のパッチ数上限は `limits.max_patches` (CUDA 向け)。CPU では `max_patches_cpu` (1024) が効くが health には出ていない。
- コンポーネントの回帰は vitest の対象外 (environment: node)。Phase 8 の Playwright E2E で担保する。
- 警告メッセージ (`warnings`) はサーバの英語のまま表示している。

---

## Phase 8: 仕上げ (M8)

- [x] エラー系の UI 網羅 (画像未選択、パッチ数超過、GPU 無し、サーバ停止中)
      (`lib/preflight.ts` が実行前に分かる失敗で「実行」を止めて理由を出す。パッチ数超過は収まる最小の patch size を添える。
      `ServerNotice` が停止中の起動コマンド + 再接続と、GPU なしの案内。`/api/match` の `NETWORK_ERROR` で health を即再取得)
- [x] `README.md`: セットアップ、起動手順、使い方、パラメータの意味、スクリーンショット (`docs/screenshot.png`)
- [x] `ruff` / `mypy` / `oxlint` / `prettier` を通す (ユーザー判断: eslint ではなく oxlint を継続し prettier を追加。CI は作らない)
      (prettier は `semi: false` / `singleQuote` / `printWidth: 120`。`npm run format` / `format:check`)
- [x] Playwright E2E (`npm run e2e`。`e2e/match.spec.ts` = アップロード → 実行 → ホバーでハイライト、
      平行移動ペアで top-1 が 1 行 2 列ずれることまで確認。`e2e/errors.spec.ts` = API をモックしたエラー系 4 本)
- [x] 性能確認: flash で end-to-end 152 ms (HTTP 往復 176 ms)。README の「性能」に記録 (`scripts/bench_api.py`)
- [x] サンプル画像を `assets/samples/` に同梱し、ワンクリックで試せるようにする
      (scikit-image 同梱の CC0 / PD 写真から `scripts/make_samples.py` で生成。`GET /api/samples`、`POST /api/samples/{id}`)
- [x] Phase 7 からの持ち越し
  - [x] flow: 「確信度で薄く」(既定 on)。確信度の分位点で不透明度 0.12〜1
  - [x] 警告の日本語化: `warnings` を `{code, message, params}` に変更 (API 変更、SPEC 4.4 に反映)。文言は `lib/warnings.ts`
  - [x] PNG 凡例: 幅が足りなければ 2 段、それでも入らなければ文字を縮める (`lib/legend.ts`)
  - [x] health の CPU 上限: 既に `limits.max_patches` が CUDA 無しで `max_patches_cpu` を返していた (実測 1024)。テストを厳密化したのみ
- [x] ついでの修正: `compute_divergence=false` でも `elapsed_ms.divergence` に同期待ちの 0.01 ms が載り、テストが不安定だった → 0 固定

**完了条件**: 初見のユーザが README だけで起動し、サンプルで動作を確認できる。
(確認済み: pytest GPU 178 passed / `CUDA_VISIBLE_DEVICES=""` で 163 passed, 15 skipped、ruff / mypy 通過。
`npm test` 104 passed、oxlint / prettier / `npm run build` 通過、E2E 5 passed。Playwright で サンプル読み込み → 実行 → フロー / pin を目視確認)

**Phase 9 以降への持ち越し**

- 既定パラメータ (`blur=0.05`) では `mirror` サンプル (N=M=768) で行質量誤差 41% の警告が出る (反復上限 512 回で停止)。
  サンプルごとの推奨パラメータを `samples.json` に持たせる案がある (未実装)。
- `左右反転` サンプルは画素特徴だとパッチ自体も反転するので、対応は「おおむね左右対称」にとどまる (フローは交差の多い横向き矢印)。
- CI は作っていない (ユーザー判断)。作るなら torch の取得が重いので lint + フロントのテストのみが現実的。

---

## Phase 9: 拡張 (オプション、優先度順)

- [x] ハードマッチングモード (`c_transform_fwd` の argmin) と UI 切替
  - 判明: 一様重みでは argmin がソフト計画の top-1 と数式上一致する (実測 100%)。ユーザー判断でバックエンド実装は継続し、
    表示量 (変位 = argmin 先の中心、受け取り量 = 選ばれた回数) の違いとして出す (SPEC 3.6)
  - `SinkhornBackend.c_transform` (flash = `c_transform_fwd`、dense = `app/ot/hard.py` のチャンク argmin)。常に計算 (約 1 ms)
  - API: `matches[].hard {j, cost, displacement}`、`hard_col_mass`、`stats.hard_agreement`、`elapsed_ms.hard`
  - UI: ModeSelector の「ソフト / ハード」。`lib/assignment.ts` の `viewResult` が結果を読み替え、overlay / flow / heatmap / PNG は共通
  - 確認: pytest GPU 187 passed / CPU 169 passed, 18 skipped、ruff / mypy。`npm test` 107 passed、tsc / oxlint / prettier / build、E2E 5 passed
- [x] Unbalanced / semi-unbalanced (`reach_x`, `reach_y`) の UI 露出と解説
  - 実測 (サンプル 3 組): 運ばれる総質量は reach=1 で 64〜97%、0.2 で 4〜74%、0.1 で 0.2〜44%。目安 0.2〜1。
    `reach_x` だけ (semi-unbalanced) では A の送出量が 1/N の最大 10 倍まで偏る
  - カバレッジを両側に (A = 送出量 row_mass、B = col_mass、色範囲は共通。ハードは B のみ)。`Heatmap.layers` に変更
  - unbalanced (`row_mass_error` が null) では flow の矢印と pin の線を送出量で薄くする (`lib/mass.ts`)。hover は数値表示のみ
  - 解説: reach_x / reach_y / compute_divergence のツールチップ、カバレッジの説明、README の実測表、SPEC 5.3
  - 確認: `npm test` 115 passed、tsc / oxlint / prettier / build、E2E 5 passed。Playwright で 別の写真 + reach 0.5 の
    カバレッジ・pin・フロー・PNG 書き出しを目視確認。バックエンドは変更なし
- [x] `apply_plan_mat_flashstyle` による O(nd) 重心射影 (チャンク実装との照合)
  - ユーザー判断: 検証用のみ (API / UI は変更なし)。`app/ot/projection.py` の `flash_barycentric_projection`
  - `mat` は `[M, d]` 必須なので `[center_B, 1, 0…]` で変位と行質量を同時に取る。`d < 3` は x/y をゼロ埋め
  - 実測: 変位の差は最大 0.02 px、行質量の相対差 < 1e-3。N=M=3072 で 7 ms (チャンク要約全体 72 ms)。
    `scripts/bench_ot` に列を追加
  - 確認: pytest GPU 196 passed / CPU 169 passed, 27 skipped、ruff / mypy。cost_scale を取り違えると照合が失敗することも確認
- [x] 深層特徴 (DINOv2 等) を `feature.type` に追加 (依存は optional extra)
  - 決定済み: 取得は `timm`、モデルは `vit_small_patch14_dinov2.lvd142m` (384 次元)、
    依存は optional extra `deep` (`uv sync --extra deep`)。未導入時は `INVALID_PARAMS` で拒否
  - `torchvision` (timm の依存) も `[tool.uv.sources]` で cu128 インデックスに固定する (torch との整合)
  - patch 14px のトークン格子は本アプリの `rows×cols` と一致しないので、
    トークンマップを格子にプーリングして `i` の対応を保つ (入力は 14 の倍数へリサイズ)
  - ユーザー判断: トークン間隔 = stride (格子の覆う領域を `(cols·14, rows·14)` にリサイズ。patch ≠ stride はパッチ中心で双線形補間)、
    384 次元そのまま (`pca_dim` は無視、normalize / position_weight は適用)、health の `feature_types` で可否を通知
  - `app/services/deep.py` (モデルはデバイスごとに遅延読み込み)。`build_features(..., images=)` で元画像を渡す。
    未導入は 422 `INVALID_PARAMS`、重みの取得失敗は 503 `INTERNAL_ERROR`
  - UI: type に dinov2 (未導入なら「(未導入)」で選択不可、選択済みなら欄のエラー + 実行を止める `lib/preflight.ts`)
  - 実測 (patch 16、ハード割当の一致率): 平行移動 pca 98% / dinov2 78%、左右反転 pca 10% / dinov2 84%。
    特徴 20〜30 ms、サーバ起動後の初回はモデル読み込みで約 2〜3 秒 (重みが未取得なら +10 秒程度)
  - 確認: pytest GPU 206 passed / CPU 178 passed, 28 skipped / extra なしで deep 系が skip、ruff / mypy。
    `npm test` 117 passed、tsc / oxlint / prettier / build、E2E 5 passed。Playwright で 左右反転 + dinov2 のフローを目視確認
  - 注意: `uv sync` だけを実行すると extra が外れる (`uv run` は外さない)
- [ ] マルチスケール (粗いパッチで絞り込み → 細かいパッチで精緻化)
- [ ] 非同期ジョブ + SSE 進捗 (大きな画像で応答が長くなる場合)
- [ ] 結果のパーマリンク共有 (パラメータと画像 ID を URL に載せる)

---

## 実装中に検証すべき前提 (要注意点)

`SPEC.md` 3.4 の内容の再掲。実装時に必ず確認すること。

1. flash-sinkhorn は **CUDA テンソル必須** — CPU 経路は dense バックエンドで担保する。
2. ポテンシャルは非 debias のものを使う (輸送計画にはこれが正しい)。
   ダイバージェンス表示用の呼び出しは別に行う。ソルバは `SamplesLoss` ではなく
   `sinkhorn_flashstyle_symmetric` を直接呼ぶ (SPEC 3.4)。
3. ε スケジューリングの **最終 ε は `blur²`**。計画再構成ではこの値を使う。
4. コスト規約: 既定 `C=||x-y||²`、`half_cost=True` で `||x-y||²/2`。
   計画再構成のコストと solver の規約を必ず一致させる。
5. 引数順は `loss(a, x, b, y)`。重みを省くと一様重みになる。
6. `potentials=True` の戻り値に autograd は付かない (本アプリでは不要)。
7. 可変サイズで繰り返し解く場合、Triton の JIT 再コンパイルが効く。`autotune=False` で
   形状ごとの再チューニングは無くなったので `pad_to_multiple` は不要 (Phase 3 で実測)。
8. **特徴のスケールと `blur` の整合** (Phase 3 で解消)。`zscore` 後に `1/√d` 倍して
   二乗距離の平均を約 2 に揃えた。スケール調整前は `blur=0.05` で行質量が 10³¹ に発散した。
