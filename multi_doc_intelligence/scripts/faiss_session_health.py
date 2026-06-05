from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DOC_REGISTRY_PATH, PATHS, ensure_storage_dirs
from indexing.vector_store import VectorStoreManager


def _registry_session_ids() -> set[str]:
    if not DOC_REGISTRY_PATH.exists():
        return set()
    try:
        payload = json.loads(DOC_REGISTRY_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return set()

    sessions = payload.get("sessions", {})
    if not isinstance(sessions, dict):
        return set()
    return {str(session_id) for session_id in sessions.keys() if str(session_id).strip()}


def _directory_session_ids() -> set[str]:
    if not PATHS.faiss_index_dir.exists():
        return set()
    return {item.name for item in PATHS.faiss_index_dir.iterdir() if item.is_dir()}


def _all_session_ids() -> list[str]:
    ids = _registry_session_ids().union(_directory_session_ids())
    return sorted(ids)


def run_health_scan(repair: bool) -> int:
    ensure_storage_dirs()

    session_ids = _all_session_ids()
    if not session_ids:
        print("[faiss-health] No local FAISS sessions found.")
        return 0

    broken_count = 0
    repaired_count = 0

    print(f"[faiss-health] Scanning {len(session_ids)} session(s). repair={repair}")
    for session_id in session_ids:
        manager = VectorStoreManager(session_id=session_id)
        health = manager.check_session_index_health()
        status = str(health.get("status", "unknown"))
        action = str(health.get("action", "none"))

        if status == "broken":
            broken_count += 1
            if repair:
                repaired = manager.repair_session_index()
                repaired_status = str(repaired.get("status", "unknown"))
                repaired_action = str(repaired.get("action", "none"))
                if repaired_status == "repaired":
                    repaired_count += 1
                print(
                    f"[faiss-health] session={session_id} status={status} action={action} "
                    f"-> repaired_status={repaired_status} repaired_action={repaired_action}"
                )
            else:
                print(f"[faiss-health] session={session_id} status={status} action={action}")
        else:
            print(f"[faiss-health] session={session_id} status={status} action={action}")

    print(
        f"[faiss-health] summary total={len(session_ids)} broken={broken_count} repaired={repaired_count}"
    )
    return 0 if broken_count == 0 or repair else 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan local FAISS session health and optionally repair broken sessions.")
    parser.add_argument(
        "--repair",
        action="store_true",
        help="Automatically repair broken FAISS sessions during scan.",
    )
    args = parser.parse_args()
    raise SystemExit(run_health_scan(repair=bool(args.repair)))


if __name__ == "__main__":
    main()
