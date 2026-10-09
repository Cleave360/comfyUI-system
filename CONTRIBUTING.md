# Contributing

Use Python 3.12 and run `./scripts/bootstrap.py --repair-links` from the repository
root. Keep upstream ComfyUI and ComfyUI-Manager changes in their own projects;
this repository owns orchestration, Jazzy, workflows, and `custom_nodes/`.

Before opening a pull request:

```bash
.venv/bin/python -m compileall -q scripts kindred-avatar/backend custom_nodes
.venv/bin/python -m pytest -q
./scripts/bootstrap.py --check --skip-deps
.venv/bin/python scripts/model_manifest.py check
zsh -n start_all.sh stop_all.sh status_all.sh
```

Never commit `.env`, `.kindred/`, model weights, generated media, logs, runtime
state, or nested upstream checkouts. New custom nodes and models must document
their source, licence, version, and platform constraints. Changes to controlled
execution must preserve the envelope, accepted-start-before-side-effect, and
durable-terminal-evidence rules.
