import asyncio
import io
import json
import logging
import time

import httpx
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from app import ot
from app.config import Settings
from app.main import create_app
from app.ot import DenseBackend
from app.services import matching
from app.services.deep import deep_available
from app.services.patches import GridMeta
from tests.conftest import encode, make_image
from tests.synth import shifted_pair, texture, true_targets


def upload(client: TestClient, data: bytes, name: str = "a.png", mime: str = "image/png"):
    return client.post("/api/images", files={"file": (name, data, mime)})


def assert_error(resp, status: int, code: str) -> dict:
    assert resp.status_code == status, resp.text
    body = resp.json()
    assert set(body) == {"detail"}
    assert set(body["detail"]) == {"code", "message", "hint"}
    assert body["detail"]["code"] == code
    return body["detail"]


def test_upload_get_roundtrip(client: TestClient):
    original = make_image(96, 48)
    resp = upload(client, encode(original))
    assert resp.status_code == 200
    body = resp.json()
    assert body["width"] == 96 and body["height"] == 48
    assert body["original_width"] == 96 and body["original_height"] == 48
    assert body["url"] == f"/api/images/{body['image_id']}"

    got = client.get(body["url"])
    assert got.status_code == 200
    assert got.headers["content-type"] == "image/png"
    fetched = Image.open(io.BytesIO(got.content))
    assert fetched.format == "PNG" and fetched.size == (96, 48)
    assert fetched.tobytes() == original.tobytes()  # 前処理で変化しない入力は無劣化


def test_upload_resizes_and_trims_to_stride(client: TestClient):
    # ノイズだと max_upload_bytes を超えるので、単色画像で寸法処理だけを見る
    resp = upload(client, encode(Image.new("RGB", (1000, 600), (10, 20, 30))))
    assert resp.status_code == 200
    body = resp.json()
    # 1000x600 -> 長辺 512 -> 512x307 -> 16 の倍数に切り詰め -> 512x304
    assert (body["width"], body["height"]) == (512, 304)
    assert (body["original_width"], body["original_height"]) == (1000, 600)
    fetched = Image.open(io.BytesIO(client.get(body["url"]).content))
    assert fetched.size == (512, 304)


def test_upload_accepts_jpeg_and_webp(client: TestClient):
    img = make_image(64, 64)
    assert upload(client, encode(img, "JPEG"), "a.jpg", "image/jpeg").status_code == 200
    assert upload(client, encode(img, "WEBP"), "a.webp", "image/webp").status_code == 200


def test_upload_too_large_413(client: TestClient):
    data = encode(make_image(300, 300))  # ノイズ PNG は max_upload_bytes (200KB) を超える
    assert len(data) > 200_000
    assert_error(upload(client, data), 413, "IMAGE_TOO_LARGE")


def test_upload_side_too_large_413(tmp_path):
    settings = Settings(
        tmp_dir=tmp_path, max_image_side=100, warmup_on_startup=False, _env_file=None
    )  # type: ignore[call-arg]
    with TestClient(create_app(settings)) as client:
        assert_error(upload(client, encode(make_image(101, 16))), 413, "IMAGE_TOO_LARGE")
        assert upload(client, encode(make_image(100, 16))).status_code == 200


@pytest.mark.parametrize(
    "data",
    [b"not an image", b"", encode(make_image(32, 32), "GIF"), encode(make_image(32, 32), "BMP")],
    ids=["text", "empty", "gif", "bmp"],
)
def test_upload_unsupported_415(client: TestClient, data: bytes):
    # Content-Type / 拡張子を偽装しても、中身で判定される
    assert_error(upload(client, data, "a.png", "image/png"), 415, "UNSUPPORTED_FORMAT")


def test_upload_truncated_png_415(client: TestClient):
    data = encode(make_image(64, 64))
    assert_error(upload(client, data[: len(data) // 2]), 415, "UNSUPPORTED_FORMAT")


def test_upload_too_small_422(client: TestClient):
    assert_error(upload(client, encode(make_image(10, 10))), 422, "INVALID_PARAMS")


def test_upload_missing_file_422(client: TestClient):
    detail = assert_error(client.post("/api/images"), 422, "INVALID_PARAMS")
    assert "file" in detail["message"]


def test_get_unknown_image_404(client: TestClient):
    assert_error(client.get("/api/images/01ARZ3NDEKTSV4RRFFQ69G5FAV"), 404, "IMAGE_NOT_FOUND")


@pytest.mark.parametrize(
    "image_id",
    ["..", "%2e%2e", "..%2f..%2fetc%2fpasswd", "01ARZ3NDEKTSV4RRFFQ69G5FAV.png", "abc", "%0a"],
)
def test_get_malformed_id_is_not_served(client: TestClient, image_id: str):
    resp = client.get(f"/api/images/{image_id}")
    assert resp.status_code == 404
    assert resp.json()["detail"]["code"] in {"IMAGE_NOT_FOUND", "HTTP_ERROR"}


def test_unknown_route_uses_error_format(client: TestClient):
    assert_error(client.get("/api/nope"), 404, "HTTP_ERROR")


def test_health(client: TestClient, settings: Settings):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert set(body) == {
        "status", "device", "cuda_available", "gpu_name", "torch_version",
        "flash_sinkhorn_version", "default_backend", "feature_types", "limits",
    }  # fmt: skip
    assert body["feature_types"][:3] == ["pca", "raw", "color"]
    assert ("dinov2" in body["feature_types"]) == deep_available()
    assert body["default_backend"] in {"flash", "dense"}
    assert body["limits"]["max_upload_bytes"] == settings.max_upload_bytes
    # CUDA が無ければ dense (CPU) 向けの上限。フロントの ParamPanel はこの値で超過を判定する
    assert body["limits"]["max_patches"] == settings.patch_limit(body["cuda_available"])


def test_cors_allows_dev_origin(client: TestClient):
    resp = client.get("/api/health", headers={"Origin": "http://localhost:5173"})
    assert resp.headers["access-control-allow-origin"] == "http://localhost:5173"
    resp = client.get("/api/health", headers={"Origin": "http://evil.example"})
    assert "access-control-allow-origin" not in resp.headers


# --- POST /api/match ---------------------------------------------------------------------------

DENSE = {"backend": "dense"}


def texture_png(t: torch.Tensor) -> bytes:
    return encode(Image.fromarray((t.permute(1, 2, 0) * 255).byte().numpy()))


def upload_id(client: TestClient, t: torch.Tensor) -> str:
    resp = upload(client, texture_png(t))
    assert resp.status_code == 200, resp.text
    return resp.json()["image_id"]


def match(client: TestClient, a: str, b: str, **sections: dict):
    return client.post("/api/match", json={"image_a": a, "image_b": b, **sections})


@pytest.fixture
def shifted_ids(client: TestClient) -> tuple[str, str]:
    """B(p) = A(p + (16, 0))。A のパッチは B の 16px 左 (2 パッチ左) に対応する。"""
    img_a, img_b = shifted_pair(96, 96, 16, 0)
    return upload_id(client, img_a), upload_id(client, img_b)


def test_match_end_to_end_dense(client: TestClient, shifted_ids: tuple[str, str]):
    resp = match(client, *shifted_ids, patch={"size": 8}, ot=DENSE)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert set(body) == {
        "grid_a", "grid_b", "matches", "col_mass", "hard_col_mass", "stats", "warnings",
    }  # fmt: skip

    grid = {"rows": 12, "cols": 12, "patch_size": 8, "stride": 8}
    assert body["grid_a"] == {**grid, "image_width": 96, "image_height": 96}
    assert body["grid_b"] == body["grid_a"]

    matches = body["matches"]
    assert [m["i"] for m in matches] == list(range(144))
    assert len(body["col_mass"]) == len(body["hard_col_mass"]) == 144
    for m in matches:
        assert set(m) == {
            "i", "confidence", "entropy", "row_mass", "displacement", "targets", "hard",
        }  # fmt: skip
        assert set(m["hard"]) == {"j", "cost", "displacement"}
        weights = [t["weight"] for t in m["targets"]]
        assert weights == sorted(weights, reverse=True) and len(weights) <= 3
        if weights:
            assert m["confidence"] == pytest.approx(weights[0])

    # 真の対応: B 側で 2 パッチ左。B の外に出る A パッチ (左端 2 列) は除く
    grid_meta = GridMeta(12, 12, 8, 8, 96, 96)
    truth = true_targets(grid_meta, grid_meta, 16, 0).tolist()
    inside = [i for i, t in enumerate(truth) if t >= 0]
    hits = sum(matches[i]["targets"][0]["j"] == truth[i] for i in inside)
    assert hits / len(inside) >= 0.9
    dx = sorted(matches[i]["displacement"][0] for i in inside)[len(inside) // 2]
    assert dx == pytest.approx(-16.0, abs=8.0)  # 中央値誤差 1 パッチ以内 (SPEC 9章)
    # ハード割当 (SPEC 3.6): b が一様なので top-1 と一致し、変位は格子上の値になる
    same = sum(m["hard"]["j"] == m["targets"][0]["j"] for m in matches)
    assert same / len(matches) == pytest.approx(body["stats"]["hard_agreement"])
    exact = sum(matches[i]["hard"]["displacement"] == [-16.0, 0.0] for i in inside)
    assert exact / len(inside) >= 0.9
    assert sum(body["hard_col_mass"]) == pytest.approx(144)  # Σ (count_j / N) / (1 / M)

    stats = body["stats"]
    assert stats["n_patches_a"] == stats["n_patches_b"] == 144
    assert stats["eps"] == pytest.approx(0.05**2)
    assert stats["backend"] == "dense"
    assert stats["sinkhorn_divergence"] > 0
    assert set(stats["elapsed_ms"]) == {
        "preprocess", "features", "solve", "divergence", "plan", "hard", "total",
    }  # fmt: skip
    assert stats["hard_agreement"] >= 0.99
    parts = sum(v for k, v in stats["elapsed_ms"].items() if k != "total")
    assert 0 < parts <= stats["elapsed_ms"]["total"] * 1.01
    assert stats["row_mass_error"] < 0.05
    # row_mass / col_mass はどちらも重みで割った値 (1.0 が期待値)
    assert max(abs(m["row_mass"] - 1) for m in matches) == pytest.approx(stats["row_mass_error"])
    assert all(abs(c - 1) < 0.1 for c in body["col_mass"])


def test_match_same_image_is_diagonal(client: TestClient):
    image_id = upload_id(client, texture(64, 64, seed=3))
    body = match(client, image_id, image_id, patch={"size": 8}, ot=DENSE).json()
    diagonal = sum(m["targets"][0]["j"] == m["i"] for m in body["matches"])
    assert diagonal / len(body["matches"]) >= 0.95


def test_match_minimal_request_uses_defaults(client: TestClient, shifted_ids: tuple[str, str]):
    resp = match(client, *shifted_ids)  # backend は省略 → サーバ設定 (auto)
    assert resp.status_code == 200, resp.text
    assert resp.json()["grid_a"]["patch_size"] == 16
    assert resp.json()["grid_a"]["stride"] == 16


def test_match_respects_output_options(client: TestClient, shifted_ids: tuple[str, str]):
    body = match(
        client,
        *shifted_ids,
        patch={"size": 8, "stride": 16},
        ot=DENSE,
        output={"top_k": 1, "min_weight": 0.999999, "compute_divergence": False},
    ).json()
    assert body["grid_a"]["rows"] == body["grid_a"]["cols"] == 6
    assert body["stats"]["sinkhorn_divergence"] is None
    assert body["stats"]["elapsed_ms"]["divergence"] == 0
    assert all(len(m["targets"]) <= 1 for m in body["matches"])
    assert any(len(m["targets"]) == 0 for m in body["matches"])  # min_weight 未満は 0 件になりうる


def test_match_unbalanced_has_no_row_mass_error(client: TestClient, shifted_ids: tuple[str, str]):
    body = match(
        client, *shifted_ids, patch={"size": 8}, ot={**DENSE, "reach_x": 1.0, "reach_y": 1.0}
    ).json()
    assert body["stats"]["row_mass_error"] is None
    assert body["stats"]["sinkhorn_divergence"] is None  # dense の unbalanced は未対応


def test_match_unknown_image_404(client: TestClient, shifted_ids: tuple[str, str]):
    ghost = "01ARZ3NDEKTSV4RRFFQ69G5FAV"
    assert_error(match(client, ghost, shifted_ids[1], ot=DENSE), 404, "IMAGE_NOT_FOUND")
    assert_error(match(client, shifted_ids[0], ghost, ot=DENSE), 404, "IMAGE_NOT_FOUND")


@pytest.mark.parametrize(
    "sections",
    [
        {"ot": {"blur": 0}},
        {"ot": {"scaling": 1.0}},
        {"ot": {"scaling": 0}},
        {"ot": {"reach_x": -1}},
        {"ot": {"backend": "gpu"}},
        {"ot": {"blurr": 0.1}},  # typo は無視せず拒否
        {"output": {"top_k": 0}},
        {"output": {"top_k": 11}},
        {"patch": {"size": 0}},
        {"feature": {"position_weight": -1}},
        {"feature": {"type": "dino"}},
    ],
    ids=lambda s: str(s),
)
def test_match_invalid_params_422(client: TestClient, sections: dict):
    detail = assert_error(match(client, "x", "y", **sections), 422, "INVALID_PARAMS")
    assert detail["message"]


def test_match_missing_fields_422(client: TestClient):
    assert_error(client.post("/api/match", json={"image_a": "x"}), 422, "INVALID_PARAMS")


def test_match_patch_larger_than_image_422(client: TestClient):
    image_id = upload_id(client, texture(32, 32, seed=1))
    detail = assert_error(
        match(client, image_id, image_id, patch={"size": 64}, ot=DENSE), 422, "INVALID_PARAMS"
    )
    assert "64" in detail["message"]


def test_match_too_many_patches_422(tmp_path):
    settings = Settings(
        tmp_dir=tmp_path, max_patches=100, max_patches_cpu=100, warmup_on_startup=False,
        _env_file=None,
    )  # type: ignore[call-arg]  # fmt: skip
    with TestClient(create_app(settings)) as client:
        img_a, img_b = shifted_pair(96, 96, 16, 0)
        a, b = upload_id(client, img_a), upload_id(client, img_b)
        detail = assert_error(
            match(client, a, b, patch={"size": 8}, ot=DENSE), 422, "TOO_MANY_PATCHES"
        )
        assert "144" in detail["message"] and "100" in detail["message"]
        # ヒントの推奨値 (stride も同じ値) で通る
        size = int(detail["hint"].split("patch size ")[1].split()[0])
        ok = match(client, a, b, patch={"size": size, "stride": size}, ot=DENSE)
        assert ok.status_code == 200, ok.text


def test_match_flash_unavailable_422(
    client: TestClient, shifted_ids: tuple[str, str], monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(ot, "_instances", {})  # 生成済みの FlashBackend を使わせない
    detail = assert_error(
        match(client, *shifted_ids, ot={"backend": "flash"}), 422, "INVALID_PARAMS"
    )
    assert "flash" in detail["message"].lower() and "dense" in detail["hint"]


class _StubBackend:
    """``DenseBackend`` に故障や待ち時間を差し込むためのスタブ。"""

    name = "stub"

    def __init__(self, *, fail: Exception | None = None, poison: bool = False, delay: float = 0):
        self._real = DenseBackend()
        self.fail, self.poison, self.delay = fail, poison, delay
        self.running = self.max_running = 0

    def solve_potentials(self, x, y, a, b, params):
        self.running += 1
        self.max_running = max(self.max_running, self.running)
        try:
            if self.fail:
                raise self.fail
            time.sleep(self.delay)
            f, g, eps, info = self._real.solve_potentials(x, y, a, b, params)
            if self.poison:
                f = f * float("nan")
            return f, g, eps, info
        finally:
            self.running -= 1

    def divergence(self, x, y, a, b, params):
        return self._real.divergence(x, y, a, b, params)

    def c_transform(self, x, y, g, params):
        return self._real.c_transform(x, y, g, params)


@pytest.mark.parametrize(
    "backend",
    [
        _StubBackend(fail=torch.cuda.OutOfMemoryError("CUDA out of memory")),
        _StubBackend(fail=RuntimeError("triton compile error")),
        _StubBackend(poison=True),
    ],
    ids=["oom", "runtime_error", "non_finite"],
)
def test_match_solver_failed_500(
    client: TestClient, shifted_ids: tuple[str, str], monkeypatch: pytest.MonkeyPatch, backend
):
    monkeypatch.setattr(matching, "get_backend", lambda _: backend)
    detail = assert_error(match(client, *shifted_ids, patch={"size": 8}), 500, "SOLVER_FAILED")
    assert detail["message"] and detail["hint"]


async def test_match_gpu_computations_do_not_overlap(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
):
    backend = _StubBackend(delay=0.15)
    monkeypatch.setattr(matching, "get_backend", lambda _: backend)
    app = create_app(settings)
    img_a, img_b = shifted_pair(64, 64, 8, 0)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
        ids = []
        for img in (img_a, img_b):
            resp = await c.post(
                "/api/images", files={"file": ("a.png", texture_png(img), "image/png")}
            )
            ids.append(resp.json()["image_id"])
        payload = {"image_a": ids[0], "image_b": ids[1], "patch": {"size": 8}}
        responses = await asyncio.gather(*(c.post("/api/match", json=payload) for _ in range(3)))
    assert [r.status_code for r in responses] == [200, 200, 200]
    assert backend.max_running == 1


def test_match_warns_when_row_mass_is_inexact(tmp_path):
    # 追加反復 0 では最終 ε で十分に収束せず、行質量の誤差が許容値を超える
    settings = Settings(
        tmp_dir=tmp_path, max_final_iters=0, warn_row_mass_error=1e-6, warmup_on_startup=False,
        _env_file=None,
    )  # type: ignore[call-arg]  # fmt: skip
    with TestClient(create_app(settings)) as client:
        img_a, img_b = shifted_pair(96, 96, 16, 0)
        body = match(
            client, upload_id(client, img_a), upload_id(client, img_b),
            patch={"size": 8}, ot=DENSE,
        ).json()  # fmt: skip
    assert body["stats"]["converged"] is False
    (w,) = [w for w in body["warnings"] if w["code"] == "ROW_MASS_ERROR"]
    assert "Row masses deviate" in w["message"]
    assert w["params"]["converged"] is False and w["params"]["error"] > w["params"]["tolerance"]


def test_match_warns_when_unbalanced_solver_hits_limit(tmp_path):
    # unbalanced では行質量が a_i からずれるのは仕様なので、代わりに収束しなかったことを警告する
    settings = Settings(
        tmp_dir=tmp_path, max_final_iters=0, warmup_on_startup=False, _env_file=None,
    )  # type: ignore[call-arg]  # fmt: skip
    with TestClient(create_app(settings)) as client:
        img_a, img_b = shifted_pair(96, 96, 16, 0)
        body = match(
            client, upload_id(client, img_a), upload_id(client, img_b),
            patch={"size": 8}, ot={**DENSE, "reach_x": 0.5, "threshold": 1e-9},
        ).json()  # fmt: skip
    assert body["stats"]["converged"] is False and body["stats"]["row_mass_error"] is None
    (w,) = body["warnings"]
    assert w["code"] == "NOT_CONVERGED"
    assert w["params"] == {"iterations": body["stats"]["iterations"]}


def test_match_warning_absent_for_benign_setup(client: TestClient, shifted_ids: tuple[str, str]):
    body = match(client, *shifted_ids, patch={"size": 8}, ot={**DENSE, "blur": 0.2}).json()
    assert body["warnings"] == []


def test_match_logs_structured_record(
    client: TestClient, shifted_ids: tuple[str, str], caplog: pytest.LogCaptureFixture
):
    with caplog.at_level(logging.INFO, logger="app.api.routes.match"):
        match(client, *shifted_ids, patch={"size": 8}, ot=DENSE)
        match(client, "bad", shifted_ids[1], ot=DENSE)
    records = [json.loads(r.getMessage().removeprefix("match ")) for r in caplog.records]
    ok, failed = records
    assert ok["status"] == "ok" and ok["image_a"] == shifted_ids[0]
    assert ok["patch"]["size"] == 8 and ok["ot"]["backend"] == "dense"
    assert {"preprocess", "features", "solve", "plan", "total"} <= set(ok["elapsed_ms"])
    assert failed["status"] == "error" and failed["code"] == "IMAGE_NOT_FOUND"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="flash backend requires CUDA")
def test_match_end_to_end_flash(client: TestClient, shifted_ids: tuple[str, str]):
    flash = match(client, *shifted_ids, patch={"size": 8}, ot={"backend": "flash"})
    dense = match(client, *shifted_ids, patch={"size": 8}, ot=DENSE)
    assert flash.status_code == dense.status_code == 200, flash.text
    fb, db = flash.json(), dense.json()
    assert fb["stats"]["backend"] == "flash" and fb["stats"]["device"].startswith("cuda")
    agree = sum(
        f["targets"][0]["j"] == d["targets"][0]["j"] for f, d in zip(fb["matches"], db["matches"])
    )
    assert agree / len(fb["matches"]) >= 0.95
    assert fb["stats"]["sinkhorn_divergence"] == pytest.approx(
        db["stats"]["sinkhorn_divergence"], rel=0.05
    )


def test_static_dir_serves_frontend_without_shadowing_api(tmp_path):
    static = tmp_path / "dist"
    static.mkdir()
    (static / "index.html").write_text("<!doctype html><title>fsw</title>")
    settings = Settings(
        tmp_dir=tmp_path, static_dir=static, warmup_on_startup=False, _env_file=None
    )  # type: ignore[call-arg]
    with TestClient(create_app(settings)) as client:
        index = client.get("/")
        assert index.status_code == 200 and "<title>fsw</title>" in index.text
        assert client.get("/api/health").status_code == 200
        assert_error(client.get("/missing.js"), 404, "HTTP_ERROR")
