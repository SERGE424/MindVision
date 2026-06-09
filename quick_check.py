#!/usr/bin/env python
import subprocess
import sys
from pathlib import Path


def run_step(title, cmd):
    print(f"\n== {title} ==")
    print("$", " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"Echec: {title} (code {result.returncode})")
        return False
    print(f"OK: {title}")
    return True


def main():
    root = Path(__file__).resolve().parent

    steps = [
        (
            "Tests conversations/ui_settings",
            [sys.executable, str(root / "test_conversations_ui_settings.py")],
        ),
        (
            "Smoke test import AssistantIA",
            [
                sys.executable,
                "-c",
                "from AssistantIA_Complet import AssistantIA; print('Import AssistantIA OK')",
            ],
        ),
    ]

    for title, cmd in steps:
        if not run_step(title, cmd):
            return 1

    print("\nVerification rapide: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
