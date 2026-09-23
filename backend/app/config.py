import tempfile
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """アプリ設定。環境変数 (接頭辞 ``FSW_``) または ``.env`` で上書きできる。"""

    model_config = SettingsConfigDict(env_prefix="FSW_", env_file=".env", extra="ignore")

    # 画像ストア
    tmp_dir: Path = Path(tempfile.gettempdir()) / "flash-sinkhorn-webui"
    image_ttl_seconds: int = Field(default=3600, gt=0)

    # アップロード制約 (SPEC 4.2)
    max_upload_bytes: int = Field(default=10 * 1024 * 1024, gt=0)
    max_image_side: int = Field(default=8192, gt=0)

    # 前処理 (SPEC 3.1)。幅・高さは ``preprocess_stride`` の倍数に切り詰める
    resize_max: int = Field(default=512, gt=0)
    preprocess_stride: int = Field(default=16, gt=0)

    # マッチング (SPEC 3.2 / 3.7)
    max_patches: int = Field(default=4096, gt=0)
    max_patches_cpu: int = Field(default=1024, gt=0)
    default_backend: Literal["auto", "flash", "dense"] = "auto"
    # ε スケジュール後の最終 ε での追加反復の上限 (API には出さない)。SPEC 3.4
    max_final_iters: int = Field(default=500, ge=0)
    # balanced のとき、行質量の相対誤差 max|row_mass/a - 1| がこれを超えたら warnings に載せる
    warn_row_mass_error: float = Field(default=0.05, gt=0)
    # 起動時に flash の Triton カーネルを事前コンパイルする (キャッシュ無しで約 8 秒)
    warmup_on_startup: bool = True

    # 同梱サンプル (samples.json と画像)。GET /api/samples
    samples_dir: Path = Path(__file__).resolve().parents[2] / "assets" / "samples"

    # ビルド済みフロントエンド (frontend/dist)。指定すると / で配信する (Docker イメージで使う)
    static_dir: Path | None = None

    # 開発時のフロントエンド (Vite dev server)
    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]

    def patch_limit(self, cuda_available: bool) -> int:
        """A・B それぞれのパッチ数の上限。CUDA が無いときは dense (CPU) 向けの小さい値。"""
        return self.max_patches if cuda_available else self.max_patches_cpu

    @property
    def image_dir(self) -> Path:
        return self.tmp_dir / "images"
