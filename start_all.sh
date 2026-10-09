#!/usr/bin/env zsh
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-${0:A:h}}"
exec "$PROJECT_ROOT/.venv/bin/python" "$PROJECT_ROOT/scripts/stack_manager.py" start
