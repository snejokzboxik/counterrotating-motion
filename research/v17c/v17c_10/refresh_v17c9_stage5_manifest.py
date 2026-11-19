"""Mark the resumed V17C.9 Sobol checkpoint complete and refresh changed hashes."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
V17C9 = HERE.parent / "v17c_9"
MANIFEST = V17C9 / "optimization" / "stage5_candidate52" / "MANIFEST.json"


def main():
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    data["status"] = "completed after explicit V17C.10 authorization; 3,120/3,120 trials valid"
    data["completed_trials"] = 3120
    data["planned_trials"] = 3120
    data["complete_config_ids"] = list(range(65))
    data["partial_config_id"] = None
    data["resumption"] = {
        "authorized_in": "research/v17c/v17c_10",
        "base_commit": "cc41be9082c62e9b907eb459f453ac51001d4267",
        "prior_completed_rows": 849,
        "prior_runs_prefix_sha256": "dd2f3295915a4137e07b496f868b5cebf3856d4befd91980e91965193c0f3b3c",
        "new_rows": 2271,
        "anchor_config_id": 64,
    }
    for item in data["files"]:
        path = V17C9 / item["path"]
        if path.is_file():
            raw = path.read_bytes()
            item["size_bytes"] = len(raw)
            item["sha256"] = hashlib.sha256(raw).hexdigest()
    MANIFEST.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"path": str(MANIFEST), "status": data["status"], "files": len(data["files"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
