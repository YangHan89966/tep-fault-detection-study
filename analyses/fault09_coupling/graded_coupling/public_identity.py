"""Verify the public distribution against the unchanged historical code lock.

Exactly one frozen test has a distribution-only path edit. Its historical hash
and its public bytes are both checked through PUBLIC_DERIVATIONS.json. All other
frozen files retain their original SHA-256 identities.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

ORIGINAL_CODE_LOCK_SHA256 = "cde66c86832bc7253c295cd28022837ec5ea26ed5f2d4ac56b7f889c6657b22b"
PATH_ONLY_FILE = "audit/test_coupling_synthetic.py"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def safe_file(root, relative):
    path = root / relative
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root):
        raise ValueError("Missing or unsafe public file: " + relative)
    return path


def verify_frozen_files(root):
    root = Path(root).resolve(strict=True)
    lock_path = safe_file(root, "CODE_LOCK.json")
    if sha(lock_path) != ORIGINAL_CODE_LOCK_SHA256:
        raise ValueError("Historical code lock identity differs")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    changes = json.loads(safe_file(root, "PUBLIC_DERIVATIONS.json").read_text(encoding="utf-8"))
    if changes["original_code_lock_sha256"] != ORIGINAL_CODE_LOCK_SHA256:
        raise ValueError("Public derivation is bound to a different historical code lock")
    mapped = changes["frozen_file_derivations"]
    if set(mapped) != {PATH_ONLY_FILE}:
        raise ValueError("Only the documented synthetic-test path edit is allowed")
    for name, original_hash in lock["files"].items():
        actual_hash = sha(safe_file(root, name))
        if name in mapped:
            record = mapped[name]
            if record["original_sha256"] != original_hash:
                raise ValueError("Historical mapped identity differs: " + name)
            if actual_hash != record["public_sha256"]:
                raise ValueError("Public mapped content differs: " + name)
        elif actual_hash != original_hash:
            raise ValueError("Frozen file changed: " + name)
    return {"status": "PASS", "historical_code_lock_sha256": ORIGINAL_CODE_LOCK_SHA256,
            "frozen_files_checked": len(lock["files"]),
            "unchanged_frozen_files": len(lock["files"]) - len(mapped),
            "documented_path_only_derivations": len(mapped)}


def verify_public_files(root):
    root = Path(root).resolve(strict=True)
    manifest = json.loads(safe_file(root, "PUBLIC_RELEASE_MANIFEST.json").read_text(encoding="utf-8"))
    for name, record in manifest["files"].items():
        path = safe_file(root, name)
        if path.stat().st_size != record["size_bytes"] or sha(path) != record["sha256"]:
            raise ValueError("Public release file differs: " + name)
    frozen = verify_frozen_files(root)
    return {"status": "PASS", "release_files_checked": len(manifest["files"]), "historical_identity": frozen}
