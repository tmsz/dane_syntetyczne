#!/usr/bin/env bash
# Because this repo lives under ~/Desktop, iCloud Drive sync sets the macOS
# "hidden" (UF_HIDDEN) flag on files inside .venv, including the editable-install
# .pth file that puts src/ on sys.path. Python's site module silently skips any
# .pth file with that flag set, so `import synthdata` fails with ModuleNotFoundError
# even though the package is installed. Clearing the flag fixes it.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
venv_dir="$repo_root/.venv"

if [ ! -d "$venv_dir" ]; then
    echo "No .venv found at $venv_dir — run 'uv sync' first." >&2
    exit 1
fi

echo "Clearing macOS hidden flags under $venv_dir ..."
chflags -R nohidden "$venv_dir"
echo "Done."
