import io
import json
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from app.config import Settings
from app.main import create_app
from app.services.samples import load_samples
from tests.conftest import encode, make_image


def test_bundled_samples_are_listed(client: TestClient, settings: Settings):
    body = client.get("/api/samples").json()
    assert [s["id"] for s in body] == [s.id for s in load_samples(settings.samples_dir)]
    assert len(body) >= 1 and set(body[0]) == {"id", "title", "description"}


def test_loading_a_sample_registers_both_images(client: TestClient):
    sample_id = client.get("/api/samples").json()[0]["id"]
    body = client.post(f"/api/samples/{sample_id}").json()
    for side in ("a", "b"):
        img = body[side]
        resp = client.get(img["url"])
        assert resp.status_code == 200
        assert Image.open(io.BytesIO(resp.content)).size == (img["width"], img["height"])
    assert body["a"]["image_id"] != body["b"]["image_id"]


def test_unknown_sample_404(client: TestClient):
    resp = client.post("/api/samples/nope")
    assert resp.status_code == 404
    assert resp.json()["detail"]["code"] == "HTTP_ERROR"


def _settings(tmp_path: Path, samples_dir: Path) -> Settings:
    return Settings(
        tmp_dir=tmp_path / "tmp", samples_dir=samples_dir, warmup_on_startup=False, _env_file=None
    )  # type: ignore[call-arg]


def test_missing_manifest_gives_empty_list(tmp_path: Path):
    with TestClient(create_app(_settings(tmp_path, tmp_path / "none"))) as c:
        assert c.get("/api/samples").json() == []


def test_broken_entries_are_skipped(tmp_path: Path):
    d = tmp_path / "samples"
    d.mkdir()
    (d / "x.png").write_bytes(encode(make_image(64, 64)))
    entries = [
        {"id": "ok", "title": "ok", "a": "x.png", "b": "x.png"},
        {"id": "missing", "title": "missing", "a": "x.png", "b": "nope.png"},
        {"id": "escape", "title": "escape", "a": "../x.png", "b": "x.png"},
    ]
    (d / "samples.json").write_text(json.dumps(entries))
    with TestClient(create_app(_settings(tmp_path, d))) as c:
        assert [s["id"] for s in c.get("/api/samples").json()] == ["ok"]
        assert c.post("/api/samples/escape").status_code == 404
