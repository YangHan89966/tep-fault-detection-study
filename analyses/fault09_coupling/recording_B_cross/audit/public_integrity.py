"""Check public bytes and explicitly recorded publication metadata transformations."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_public_bundle(root):
    root = Path(root).resolve()
    manifest = json.loads((root / "PUBLIC_MANIFEST.json").read_text())
    for name, expected in manifest["files"].items():
        p = root / name
        assert p.is_file() and not p.is_symlink(), name
        assert p.resolve().is_relative_to(root), name
        assert digest(p) == expected, "Public file differs: " + name
    changes = json.loads((root / "PUBLICATION_CHANGES.json").read_text())
    assert digest(root / "results/FINAL_LOCK.json") == changes["historical_result_final_lock_sha256"]
    edited = {}
    for entry in changes["changed_files"]:
        name = entry["public_path"]
        assert not name.endswith((".csv", ".npz")), "Scientific numerical files cannot be transformed"
        assert name not in edited, name
        assert digest(root / name) == entry["public_sha256"], name
        edited[name] = entry
    projected = json.loads((root / "reference/FINAL_LOCK.json").read_text())
    assert projected["publication_projection"]["source_document_sha256"] == edited["reference/FINAL_LOCK.json"]["original_sha256"]
    return edited, len(manifest["files"])


def check_historical_digest(root, path, expected, edited):
    actual = digest(path)
    if actual == expected:
        return
    name = Path(path).resolve().relative_to(Path(root).resolve()).as_posix()
    entry = edited.get(name)
    assert name == "results/INPUT_PREFLIGHT.json", "Unexpected historical digest mismatch: " + name
    assert entry is not None and entry["original_sha256"] == expected and entry["public_sha256"] == actual, name
