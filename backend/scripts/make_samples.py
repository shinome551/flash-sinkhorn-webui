"""scikit-image 同梱の写真 (CC0 / パブリックドメイン) から ``assets/samples/`` のサンプルを作る。

scikit-image はアプリの依存に入れていないので、一時的に足して実行する::

    cd backend
    uv run --with scikit-image python -m scripts.make_samples

画像と ``samples.json`` (``GET /api/samples`` の一覧) を上書きする。
"""

import json
from pathlib import Path

import numpy as np
from PIL import Image
from skimage import data  # type: ignore[import-not-found]

OUT = Path(__file__).resolve().parents[2] / "assets" / "samples"


def _save(arr: np.ndarray, name: str) -> str:
    Image.fromarray(arr).save(OUT / name, quality=92)
    return name


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    # 1. 平行移動: 同じ写真から (dx, dy) = (32, 16) ずらして切り出す。flow が一様な右下向きになる
    cat = data.chelsea()  # 300×451
    w, h, dx, dy = 400, 272, 32, 16
    shift_a = _save(cat[dy : dy + h, dx : dx + w], "cat_a.jpg")
    shift_b = _save(cat[:h, :w], "cat_b.jpg")

    # 2. 左右反転: 512×384 (patch 16 で N = M = 768。SPEC 8 章の性能目標と同じ規模)
    astronaut = data.astronaut()[:384]
    flip_a = _save(astronaut, "astronaut.jpg")
    flip_b = _save(np.ascontiguousarray(astronaut[:, ::-1]), "astronaut_mirror.jpg")

    # 3. 別の写真: 色の近いパッチどうしが対応する。coverage で受け取りの偏りが見える
    coffee = np.asarray(Image.fromarray(data.coffee()).resize((480, 320), Image.Resampling.LANCZOS))
    other_a = _save(coffee, "coffee.jpg")
    other_b = _save(cat[:288, :432], "cat.jpg")

    samples = [
        {
            "id": "shift",
            "title": "平行移動",
            "description": "同じ写真を右下へ (32, 16) px ずらした 2 枚。フローモードで一様な矢印になる。",
            "a": shift_a,
            "b": shift_b,
        },
        {
            "id": "mirror",
            "title": "左右反転",
            "description": "512×384 の写真とその鏡像 (patch 16 で N = M = 768)。対応先が左右対称の位置になる。",
            "a": flip_a,
            "b": flip_b,
        },
        {
            "id": "different",
            "title": "別の写真",
            "description": "コーヒーと猫。色や模様の近いパッチどうしが対応し、カバレッジで受け取りの偏りが見える。",
            "a": other_a,
            "b": other_b,
        },
    ]
    (OUT / "samples.json").write_text(json.dumps(samples, ensure_ascii=False, indent=2) + "\n")
    print(f"wrote {len(samples)} samples to {OUT}")


if __name__ == "__main__":
    main()
