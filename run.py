"""Backend-only launcher.

Usage:
  python run.py
  python run.py --host 0.0.0.0 --port 8000
  python run.py --no-reload
  python run.py --install
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import NoReturn


BACKEND_DIR = Path(__file__).resolve().parent


def _die(msg: str, code: int = 2) -> NoReturn:
    print(f"error: {msg}", file=sys.stderr)
    raise SystemExit(code)


def _run(cmd: list[str], *, cwd: Path, check: bool = True) -> int:
    proc = subprocess.run(cmd, cwd=str(cwd), check=False)
    if check and proc.returncode != 0:
        _die(f"command failed ({proc.returncode}): {' '.join(cmd)}", code=proc.returncode)
    return proc.returncode


def _install_backend_requirements() -> None:
    req = BACKEND_DIR / "requirements.txt"
    if not req.exists():
        _die(f"missing requirements file: {req}")
    _run([sys.executable, "-m", "pip", "install", "-r", str(req)], cwd=BACKEND_DIR)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run only the backend API server.")
    parser.add_argument("--host", default="127.0.0.1", help="backend host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="backend port (default: 8000)")
    parser.add_argument(
        "--no-reload",
        action="store_true",
        help="disable uvicorn autoreload (default: enabled)",
    )
    parser.add_argument(
        "--install",
        action="store_true",
        help="install backend Python dependencies before starting",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    if os.getenv("VERCEL") or os.getenv("VERCEL_ENV") or os.getenv("NOW_REGION"):
        print("Detected Vercel environment; skipping local dev server startup.")
        return 0

    args = parse_args(argv)

    if args.install:
        _install_backend_requirements()

    try:
        import uvicorn  # noqa: F401
    except Exception:
        _die("uvicorn is not installed. Run: python run.py --install")

    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "main:app",
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]
    if not args.no_reload:
        cmd.append("--reload")

    print(f"backend: http://{args.host}:{args.port}")
    print(f"backend docs: http://{args.host}:{args.port}/docs")
    _run(cmd, cwd=BACKEND_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

