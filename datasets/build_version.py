from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


def repo_root() -> Path:
    # datasets/build_version.py -> repo root is parent of "datasets"
    return Path(__file__).resolve().parents[1]


def sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def read_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def write_lines(path: Path, lines: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for line in lines:
            f.write(line + "\n")


def resolve_paths(paths: List[str], root: Path, base_dir: Path) -> List[Path]:
    out: List[Path] = []
    for p in paths:
        pp = Path(p)
        if not pp.is_absolute():
            pp = (base_dir / pp).resolve()
        out.append(pp)
    return out


def validate_manifest(m: Dict[str, Any]) -> None:
    if "version" not in m or not isinstance(m["version"], str) or not m["version"]:
        raise ValueError("Manifest must contain non-empty string field: version")

    if "classes" not in m or not isinstance(m["classes"], list) or not m["classes"]:
        raise ValueError("Manifest must contain non-empty list field: classes")

    if "splits" in m:
        m["lists"] = m["splits"]
    elif "lists" in m:
        m["splits"] = m["lists"]
    else:
        raise ValueError("Manifest must contain dict field: splits")
    if not isinstance(m["splits"], dict):
        raise ValueError("Manifest field splits must be a dict")

    for split in ("train", "val", "test"):
        if split not in m["splits"]:
            # allow missing splits -> treat as empty
            m["splits"][split] = []
        if not isinstance(m["splits"][split], list):
            raise ValueError(f"Manifest splits.{split} must be a list")


def build_yolo_data_yaml(
    out_dir: Path,
    train_txt: Path,
    val_txt: Path,
    classes: List[str],
) -> Path:
    yolo_dir = out_dir / "yolo"
    yolo_dir.mkdir(parents=True, exist_ok=True)

    data_yaml_path = yolo_dir / "data.yaml"
    payload = {
        "train": str(train_txt.resolve()),
        "val": str(val_txt.resolve()),
        "nc": len(classes),
        "names": classes,  # Ultralytics accepts list of class names
    }
    with data_yaml_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False)
    return data_yaml_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Build dataset version artifacts from a manifest (v0: explicit splits).")
    ap.add_argument("--version", type=str, required=False, help="Dataset version key, e.g. v0")
    ap.add_argument("--manifest", type=str, required=False, help="Path to manifest YAML. Overrides --version.")
    ap.add_argument("--pool-root", type=str, default="data/pool", help="Base directory for relative paths.")
    ap.add_argument("--allow-missing", action="store_true", help="Do not fail if listed files are missing.")
    args = ap.parse_args()

    root = repo_root()
    pool_root = (root / args.pool_root).resolve()

    if args.manifest:
        manifest_path = Path(args.manifest)
        if not manifest_path.is_absolute():
            manifest_path = (root / manifest_path).resolve()
    else:
        if not args.version:
            raise SystemExit("Provide either --version or --manifest")
        manifest_path = (root / "data" / "manifests" / f"dataset_{args.version}.yaml").resolve()

    if not manifest_path.exists():
        raise SystemExit(f"Manifest not found: {manifest_path}")

    raw_bytes = manifest_path.read_bytes()
    manifest_hash = sha256_bytes(raw_bytes)

    m = read_yaml(manifest_path)
    validate_manifest(m)

    version = m["version"]
    classes: List[str] = m["classes"]

    splits = m["splits"]
    train_list = resolve_paths(splits.get("train", []), root, pool_root)
    val_list = resolve_paths(splits.get("val", []), root, pool_root)
    test_list = resolve_paths(splits.get("test", []), root, pool_root)

    if not args.allow_missing:
        missing = [p for p in (train_list + val_list + test_list) if not p.exists()]
        if missing:
            msg = "\n".join(str(p) for p in missing[:30])
            more = "" if len(missing) <= 30 else f"\n... and {len(missing) - 30} more"
            raise SystemExit(f"Missing files listed in manifest (use --allow-missing to ignore):\n{msg}{more}")

    out_dir = (root / "data" / "versions" / version).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Write split lists (absolute paths)
    split_dir = out_dir / "split"
    train_txt = split_dir / "train.txt"
    val_txt = split_dir / "val.txt"
    test_txt = split_dir / "test.txt"

    write_lines(train_txt, [str(p.resolve()) for p in train_list])
    write_lines(val_txt, [str(p.resolve()) for p in val_list])
    write_lines(test_txt, [str(p.resolve()) for p in test_list])

    # YOLO artifacts
    data_yaml = build_yolo_data_yaml(out_dir, train_txt, val_txt, classes)

    # Build metadata
    meta = {
        "version": version,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_hash,
        "pool_root": str(pool_root),
        "counts": {
            "train": len(train_list),
            "val": len(val_list),
            "test": len(test_list),
        },
        "artifacts": {
            "train_txt": str(train_txt),
            "val_txt": str(val_txt),
            "test_txt": str(test_txt),
            "yolo_data_yaml": str(data_yaml),
        },
    }
    meta_path = out_dir / "build_meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] Built dataset version: {version}")
    print(f"  train={len(train_list)} val={len(val_list)} test={len(test_list)}")
    print(f"  out: {out_dir}")
    print(f"  yolo: {data_yaml}")


if __name__ == "__main__":
    main()
