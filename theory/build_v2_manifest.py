"""Build a SHA-256 manifest for the current V2 research branch delta."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def changed_paths() -> list[str]:
    changed = set()
    for command in (("git", "diff", "--name-only", "HEAD"),
                    ("git", "ls-files", "--others", "--exclude-standard")):
        out = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True).stdout
        changed.update(p.replace("\\", "/") for p in out.splitlines() if p)
    return sorted(p for p in changed if p != "MANIFEST.json" and p != "MANIFEST.sha256")


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def build() -> dict:
    files = []
    for rel in changed_paths():
        path = ROOT / rel
        if path.is_file():
            files.append({"path": rel, "bytes": path.stat().st_size,
                          "sha256": file_sha(path)})
    return {
        "schema": "counterrotating-motion-v2-manifest-1",
        "branch": "research/three-parameter-counterrotation-theory",
        "base_commit": "97dbf1cd004580a7ec7c698ffe94ad764795da8b",
        "source_references": {
            "V17C20B": "d1bf252649e174166becf464269318b5f97acd5d",
            "V17C21A": "94b8edfaac8a1f0b3eb518ea4c7594a80d47daf8",
            "V17C21B": "894c6896ce5927371708037ffa4a4021189d2b54"},
        "test_command": "python -m pytest -q tests",
        "test_result": "11 passed",
        "latex_build_status": "UNVERIFIED_BUILD: pdfLaTeX not installed; Tectonic not substituted",
        "full_rf_new_integrations": 0,
        "scientific_status": "M0 zero-speed slow-torque sign criterion established under stated assumptions; no full-RF attractor or exact boundary claimed",
        "files": files,
        "manifest_note": "This manifest excludes its own hash; see MANIFEST.sha256 for the manifest file checksum."}


if __name__ == "__main__":
    (ROOT / "MANIFEST.json").write_text(
        json.dumps(build(), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
