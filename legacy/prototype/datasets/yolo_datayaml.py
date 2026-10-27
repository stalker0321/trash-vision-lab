"""Generates a YOLO data.yaml from version artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


def repo_root() -> Path:
    # Resolve the repository containing legacy/prototype/datasets.
    return Path(__file__).resolve().parents[3]


def read_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def read_classes(manifest_path: Path) -> List[str]:
    manifest = read_yaml(manifest_path)
    classes = manifest.get("classes", [])
    if not isinstance(classes, list) or not classes:
        raise SystemExit(f"Manifest missing non-empty classes list: {manifest_path}")
    return classes


def manifest_from_meta(version_dir: Path) -> Optional[Path]:
    meta_path = version_dir / "build_meta.json"
    if not meta_path.exists():
        return None
    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid build_meta.json: {meta_path}") from exc
    manifest_path = payload.get("manifest_path")
    if not isinstance(manifest_path, str) or not manifest_path:
        return None
    return Path(manifest_path)


def build_data_yaml(version: str, root: Optional[Path] = None, out_path: Optional[Path] = None) -> Path:
    root = root or repo_root()
    version_dir = (root / "data" / "versions" / version).resolve()
    if not version_dir.exists():
        raise SystemExit(f"Version directory not found: {version_dir}")

    split_dir = version_dir / "split"
    train_txt = split_dir / "train.txt"
    val_txt = split_dir / "val.txt"
    if not train_txt.exists() or not val_txt.exists():
        raise SystemExit(f"Missing split lists in {split_dir}. Run datasets.build_version first.")

    manifest_path = manifest_from_meta(version_dir)
    if manifest_path is None:
        manifest_path = (root / "legacy" / "prototype" / "data" / "manifests" / f"dataset_{version}.yaml").resolve()
    if not manifest_path.exists():
        raise SystemExit(f"Manifest not found: {manifest_path}")

    classes = read_classes(manifest_path)

    if out_path is None:
        out_path = version_dir / "yolo" / "data.yaml"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "train": str(train_txt.resolve()),
        "val": str(val_txt.resolve()),
        "nc": len(classes),
        "names": classes,
    }
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate a YOLO data.yaml from built version artifacts.")
    ap.add_argument("--version", required=True, help="Dataset version key, e.g. v0")
    ap.add_argument("--out", default=None, help="Optional output path for data.yaml")
    args = ap.parse_args()

    out_path = Path(args.out).resolve() if args.out else None
    data_yaml = build_data_yaml(args.version, out_path=out_path)
    print(f"[OK] Wrote: {data_yaml}")


if __name__ == "__main__":
    main()
