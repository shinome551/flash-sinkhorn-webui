"""DINOv2 特徴 (``feature.type = "dinov2"``)。extra ``deep`` が無ければ大半を skip する。

初回は重み (約 90 MB) を Hugging Face Hub から取得する。
"""

from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from app.services import deep, matching
from app.services.features import FeatureParams, build_features
from app.services.patches import GridMeta, extract_patches, image_to_tensor

needs_deep = pytest.mark.skipif(not deep.deep_available(), reason="extra 'deep' not installed")
DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"))]  # fmt: skip
SAMPLES = Path(__file__).resolve().parents[2] / "assets" / "samples"


def load(name: str) -> torch.Tensor:
    return image_to_tensor(Image.open(SAMPLES / name).convert("RGB"))


def features(img_a, img_b, patch: int, stride: int, device="cpu", **kw):
    pa, ga = extract_patches(img_a, patch, stride)
    pb, gb = extract_patches(img_b, patch, stride)
    params = FeatureParams(type="dinov2", **kw)
    return build_features(pa, ga, pb, gb, params, device, images=(img_a, img_b)), ga, gb


@needs_deep
@pytest.mark.parametrize("device", DEVICES)
def test_tokens_align_with_patches_on_shifted_pair(device):
    """A(p) = B(p + (32, 16))。patch 16 なら A の (r, c) は B の (r+1, c+2) に対応する。"""
    img_a, img_b = load("cat_a.jpg"), load("cat_b.jpg")
    f, ga, gb = features(img_a, img_b, 16, 16, device, normalize="l2")
    assert f.x.shape == (ga.n, deep.DIM) and f.y.shape == (gb.n, deep.DIM)
    assert f.x.device.type == device and f.x.dtype == torch.float32

    nn = (f.x @ f.y.t()).argmax(dim=1).cpu()
    r, c = torch.arange(ga.n) // ga.cols, torch.arange(ga.n) % ga.cols
    inside = (r + 1 < gb.rows) & (c + 2 < gb.cols)
    truth = (r + 1) * gb.cols + (c + 2)
    assert (nn[inside] == truth[inside]).float().mean() >= 0.8


@needs_deep
def test_patch_equal_stride_is_token_map():
    """patch == stride ではトークン (r, c) がそのままパッチ i = r * cols + c になる。"""
    img = load("coffee.jpg")[:, :96, :128]
    grid = GridMeta(rows=6, cols=8, patch_size=16, stride=16, image_width=128, image_height=96)
    got = deep.dinov2_features(img, grid)

    model = deep._get_model(torch.device("cpu"))
    x = torch.nn.functional.interpolate(
        img.unsqueeze(0), size=(6 * 14, 8 * 14), mode="bilinear", antialias=True
    )
    mean = torch.tensor(model.pretrained_cfg["mean"]).view(1, 3, 1, 1)
    std = torch.tensor(model.pretrained_cfg["std"]).view(1, 3, 1, 1)
    with torch.no_grad():
        tokens = model.forward_features((x - mean) / std)[0, model.num_prefix_tokens :]
    torch.testing.assert_close(got, tokens)


@needs_deep
@pytest.mark.parametrize(("patch", "stride"), [(24, 16), (8, 16), (16, 12)])
def test_overlapping_and_sparse_grids(patch, stride):
    img = load("coffee.jpg")[:, :100, :130]
    f, ga, _ = features(img, img, patch, stride, normalize="zscore", position_weight=1.0)
    assert f.x.shape == (ga.n, deep.DIM + 2)
    assert torch.isfinite(f.x).all()
    torch.testing.assert_close(f.x, f.y)


def test_requires_images():
    pa, ga = extract_patches(torch.rand(3, 32, 32), 16, 16)
    with pytest.raises(ValueError, match="images"):
        build_features(pa, ga, pa, ga, FeatureParams(type="dinov2"))


# --- API ---------------------------------------------------------------------------------------


def upload_sample(client: TestClient, name: str) -> str:
    data = (SAMPLES / name).read_bytes()
    resp = client.post("/api/images", files={"file": (name, data, "image/jpeg")})
    assert resp.status_code == 200, resp.text
    return resp.json()["image_id"]


@needs_deep
def test_match_dinov2(client: TestClient):
    a, b = upload_sample(client, "cat_a.jpg"), upload_sample(client, "cat_b.jpg")
    body = {"image_a": a, "image_b": b, "feature": {"type": "dinov2"}, "ot": {"blur": 0.1}}
    resp = client.post("/api/match", json=body)
    assert resp.status_code == 200, resp.text
    result = resp.json()
    assert result["stats"]["feature_dim"] == deep.DIM
    # A(p) = B(p + (32, 16))。変位の中央値が真値に近い
    dx = sorted(m["hard"]["displacement"][0] for m in result["matches"])
    dy = sorted(m["hard"]["displacement"][1] for m in result["matches"])
    assert dx[len(dx) // 2] == pytest.approx(32, abs=8)
    assert dy[len(dy) // 2] == pytest.approx(16, abs=8)


def test_match_dinov2_unavailable_422(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(matching, "deep_available", lambda: False)
    a = upload_sample(client, "cat_a.jpg")
    resp = client.post(
        "/api/match", json={"image_a": a, "image_b": a, "feature": {"type": "dinov2"}}
    )
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["code"] == "INVALID_PARAMS" and "--extra deep" in detail["hint"]


def test_match_dinov2_load_failure_503(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    def fail(_):
        raise deep.ModelLoadError("LocalEntryNotFoundError: offline")

    monkeypatch.setattr(matching, "deep_available", lambda: True)
    monkeypatch.setattr(deep, "_get_model", fail)
    a = upload_sample(client, "cat_a.jpg")
    resp = client.post(
        "/api/match", json={"image_a": a, "image_b": a, "feature": {"type": "dinov2"}}
    )
    assert resp.status_code == 503
    detail = resp.json()["detail"]
    assert detail["code"] == "INTERNAL_ERROR" and "offline" in detail["message"]
