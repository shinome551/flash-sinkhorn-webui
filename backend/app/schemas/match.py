"""``POST /api/match`` のリクエスト/レスポンス (SPEC 4.4)。

リクエストは ``extra="forbid"`` (パラメータ名の typo を黙って無視しない)。範囲外は Pydantic が
検出し、``INVALID_PARAMS`` (422) として返る。
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class _Request(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PatchRequest(_Request):
    size: int = Field(default=16, ge=2, le=64)
    stride: int | None = Field(default=None, ge=1, description="省略時は size")


class FeatureRequest(_Request):
    type: Literal["raw", "pca", "color", "dinov2"] = "pca"
    pca_dim: int = Field(default=64, ge=1, le=256)
    normalize: Literal["none", "zscore", "l2"] = "zscore"
    position_weight: float = Field(default=0.0, ge=0, le=100)


class OTRequest(_Request):
    blur: float = Field(default=0.05, gt=0, le=10)
    scaling: float = Field(default=0.5, gt=0, lt=1)
    half_cost: bool = False
    reach_x: float | None = Field(default=None, gt=0)
    reach_y: float | None = Field(default=None, gt=0)
    threshold: float | None = Field(default=1e-3, gt=0, description="ε 相対。null で打ち切りなし")
    inner_iterations: int = Field(default=10, ge=1, le=1000)
    backend: Literal["auto", "flash", "dense"] | None = Field(
        default=None, description="省略時はサーバ設定 (default_backend)"
    )


class OutputRequest(_Request):
    top_k: int = Field(default=3, ge=1, le=10)
    min_weight: float = Field(default=1e-4, ge=0, lt=1)
    compute_divergence: bool = True


class MatchRequest(_Request):
    image_a: str
    image_b: str
    patch: PatchRequest = Field(default_factory=PatchRequest)
    feature: FeatureRequest = Field(default_factory=FeatureRequest)
    ot: OTRequest = Field(default_factory=OTRequest)
    output: OutputRequest = Field(default_factory=OutputRequest)


class GridInfo(BaseModel):
    rows: int
    cols: int
    patch_size: int
    stride: int
    image_width: int
    image_height: int


class Target(BaseModel):
    j: int
    weight: float  # 条件付き確率 p̃_ij
    cost: float


class HardTarget(BaseModel):
    """c-transform の argmin ``j* = argmin_j [C_ij - g_j]`` (SPEC 3.6)。"""

    j: int
    cost: float  # C_ij*
    displacement: tuple[float, float]  # center_B(j*) - center_A(i)


class PatchMatch(BaseModel):
    i: int
    confidence: float
    entropy: float
    row_mass: float
    displacement: tuple[float, float]  # ピクセル単位 (dx, dy)
    targets: list[Target]
    hard: HardTarget


class MatchStats(BaseModel):
    n_patches_a: int
    n_patches_b: int
    feature_dim: int
    eps: float
    sinkhorn_divergence: float | None
    backend: str
    device: str
    iterations: int
    converged: bool
    row_mass_error: float | None  # balanced のときの max|row_mass/a - 1|
    hard_agreement: float  # hard.j がソフトの top-1 と一致した割合 (b が一様なら ≈ 1)
    elapsed_ms: dict[str, float]


class MatchWarning(BaseModel):
    """``code`` は ``app.services.notices.WarningCode``。``message`` は英語の説明。"""

    code: str
    message: str
    params: dict[str, int | float | bool]


class MatchResponse(BaseModel):
    grid_a: GridInfo
    grid_b: GridInfo
    matches: list[PatchMatch]
    col_mass: list[float]
    hard_col_mass: list[float]  # ハード割当での受け取り量 (Σ_{i: j*=j} a_i) / b_j
    stats: MatchStats
    warnings: list[MatchWarning]
