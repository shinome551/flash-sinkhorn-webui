"""同梱サンプル (``assets/samples/samples.json``) の読み込み。

マニフェストは ``scripts/make_samples.py`` が書く。画像ファイル名はディレクトリ直下の名前だけを許す。
"""

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Sample:
    id: str
    title: str
    description: str
    path_a: Path
    path_b: Path


def load_samples(samples_dir: Path) -> list[Sample]:
    """マニフェストが無ければ空。壊れたエントリ (画像が無い・パスが不正) は読み飛ばす。"""
    manifest = samples_dir / "samples.json"
    if not manifest.is_file():
        return []
    samples = []
    for entry in json.loads(manifest.read_text(encoding="utf-8")):
        paths = [samples_dir / entry[k] for k in ("a", "b") if Path(entry[k]).name == entry[k]]
        if len(paths) == 2 and all(p.is_file() for p in paths):
            samples.append(
                Sample(entry["id"], entry["title"], entry.get("description", ""), *paths)
            )
    return samples
